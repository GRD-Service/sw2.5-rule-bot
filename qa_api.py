from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

import json
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage


# ============================================================
# Environment / settings
# ============================================================

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "./vector_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
BOOK_CATEGORY_PATH = Path(
    os.getenv("BOOK_CATEGORY_PATH", "./metadata/book_categories.json")
)
NAVIGATION_DIR = Path(
    os.getenv("NAVIGATION_DIR", "./metadata/navigation")
)

HYBRID_CANDIDATE_K = int(os.getenv("HYBRID_CANDIDATE_K", "100"))

SOURCE_AUTHORITY_STRENGTH = float(
    os.getenv("SOURCE_AUTHORITY_STRENGTH", "0.20")
)
SOURCE_AUTHORITY_MAX = float(os.getenv("SOURCE_AUTHORITY_MAX", "1.35"))

NAV_INDEX_MAX_PAGES = int(os.getenv("NAV_INDEX_MAX_PAGES", "8"))
NAV_TOC_MAX_PAGES = int(os.getenv("NAV_TOC_MAX_PAGES", "4"))
NAV_SECTION_EXPAND_PAGES = int(os.getenv("NAV_SECTION_EXPAND_PAGES", "2"))
NAV_SECTION_MAX_PAGES = int(os.getenv("NAV_SECTION_MAX_PAGES", "6"))
NAV_MANDATORY_MAX_PAGES = int(os.getenv("NAV_MANDATORY_MAX_PAGES", "12"))
NAV_CHUNKS_PER_PAGE = int(os.getenv("NAV_CHUNKS_PER_PAGE", "2"))
CONTEXT_MAX_DOCS = int(os.getenv("CONTEXT_MAX_DOCS", "24"))
NAV_INDEX_FUZZY_THRESHOLD = float(
    os.getenv("NAV_INDEX_FUZZY_THRESHOLD", "0.72")
)
NAV_TOC_FUZZY_THRESHOLD = float(
    os.getenv("NAV_TOC_FUZZY_THRESHOLD", "0.60")
)
CITATION_EXCERPT_CHARS = int(os.getenv("CITATION_EXCERPT_CHARS", "360"))

NAVIGATION_REQUIRED = os.getenv("NAVIGATION_REQUIRED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# ============================================================
# FAISS / documents
# ============================================================

embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)

db = FAISS.load_local(
    INDEX_DIR,
    embedding,
    allow_dangerous_deserialization=True,
)

all_index_documents = list(db.docstore._dict.values())

# logical_pageを持たない表紙・カバー等はstrict retrievalから除外する。
search_documents = [
    doc
    for doc in all_index_documents
    if doc.metadata.get("logical_page") is not None
]

page_documents_by_pdf = defaultdict(list)
for doc in search_documents:
    book = doc.metadata.get("book")
    pdf_page = doc.metadata.get("pdf_page", doc.metadata.get("page"))
    if not book or pdf_page is None:
        continue
    try:
        pdf_page = int(pdf_page)
    except (TypeError, ValueError):
        continue
    page_documents_by_pdf[(book, pdf_page)].append(doc)

for docs in page_documents_by_pdf.values():
    docs.sort(key=lambda doc: int(doc.metadata.get("chunk", 0)))


# ============================================================
# Lexical index
# ============================================================

lexical_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    min_df=1,
)
lexical_matrix = lexical_vectorizer.fit_transform(
    [doc.page_content for doc in search_documents]
)


# ============================================================
# Book categories / authority
# ============================================================

with BOOK_CATEGORY_PATH.open("r", encoding="utf-8") as f:
    category_data = json.load(f)

book_to_category = {}
category_weight = {}

for category, info in sorted(
    category_data.items(),
    key=lambda item: item[1].get("weight", 1.0),
    reverse=True,
):
    for book_entry in sorted(info.get("books", []), key=lambda x: x["name"]):
        book_to_category[book_entry["name"]] = category
    category_weight[category] = info.get("weight", 1.0)


def get_book_authority(book: str) -> float:
    category = book_to_category.get(book, "その他")
    raw_weight = float(category_weight.get(category, 1.0))
    authority = 1.0 + max(0.0, raw_weight - 1.0) * SOURCE_AUTHORITY_STRENGTH
    return min(SOURCE_AUTHORITY_MAX, authority)


def get_document_category(doc) -> str:
    book = doc.metadata.get("book")
    return doc.metadata.get(
        "category",
        book_to_category.get(book, "その他"),
    )


def get_document_authority(doc) -> float:
    return get_book_authority(doc.metadata.get("book", ""))


# ============================================================
# Navigation loading
# ============================================================


def load_navigation_data(path: Path) -> dict:
    if not path.exists():
        if NAVIGATION_REQUIRED:
            raise FileNotFoundError(f"navigation directory not found: {path}")
        return {}

    json_files = sorted(path.glob("*.json"))
    if not json_files and NAVIGATION_REQUIRED:
        raise RuntimeError(f"navigation JSONがありません: {path}")

    result = {}
    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"WARNING: navigation JSON load failed: {json_path}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        book = data.get("book")
        if book:
            result[book] = data
    return result


navigation_data = load_navigation_data(NAVIGATION_DIR)
print(f"Navigation books loaded: {len(navigation_data)}")


# ============================================================
# FastAPI / API models
# ============================================================

app = FastAPI()


class QueryRequest(BaseModel):
    question: str
    books: Optional[List[str]] = None
    model: Optional[str] = "gpt-5.4-nano"
    k: Optional[int] = 10
    mode: Optional[str] = "rules_strict"


class Citation(BaseModel):
    id: int
    book: str
    page: int
    pdf_page: int
    category: Optional[str] = None
    excerpt: Optional[str] = None
    reason: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    model_used: Optional[str] = None
    # 実際にLLM contextへ投入したchunk数
    k_used: Optional[int] = None
    # 通常Hybrid Searchから採用したchunk数
    hybrid_k_used: Optional[int] = None
    # navigationから採用したページ数
    navigation_pages_used: Optional[int] = None
    max_k: Optional[int] = None
    token_usage: Optional[dict] = None


# ============================================================
# Prompt
# ============================================================

template = """
以下のコンテキストだけを根拠として、質問に正確に答えてください。

回答ルール:
- コンテキストに存在しない情報を推測して補ってはいけません。
- 根拠を示す場合は、対応するコンテキストの引用IDを `[C1]` の形式で記載してください。
- 引用IDは必ずコンテキスト中に存在するものだけを使用してください。
- `[C1]`、`[C2]` のような形式以外で出典を書いてはいけません。
- 書籍名やページ番号を回答文中へ直接書く必要はありません。
- 出典は、その出典によって裏付けられる文章または段落の末尾に記載してください。
- 同一段落内では、同じ引用IDを繰り返してはいけません。
- 同一の引用IDに基づく内容が連続する場合は、可能な限り一つの段落にまとめ、段落末尾に引用IDを1回だけ記載してください。
- 異なる引用元へ内容が切り替わった場合は、その文章または段落の末尾に新しい引用IDを記載してください。
- 一つの段落が複数の引用元に基づく場合は、段落末尾に `[C1][C2]` のようにまとめて記載してください。
- 長い回答では、どの記述がどの根拠に基づくか判別できるよう、必要な段落ごとに引用してください。
- 「索引」や「目次」が検索根拠として使われていても、それ自体をルール本文として扱わず、対応する本文ページの内容を根拠にしてください。
- 同じ事項について複数の書籍に記載がある場合、基本ルールブックに基本的な定義・数値・種族特徴・ルール本文が存在するなら、原則としてそれを回答の基礎にしてください。
- サプリメントや追加書籍の記述は、基本ルールを置き換えるものと明記されていない限り、追加情報・補足情報として扱ってください。
- 質問が希少種、追加種族、追加技能、追加魔法、追加アイテム、追加戦闘特技など、特定の追加要素を明示している場合は、その要素を収録したサプリメント側の記述を優先してください。
- 基本種と希少種、基本ルールと追加ルールなど、異なる対象を混同しないでください。
- 基本資料と追加資料で説明内容が異なる場合、コンテキストから変更・追加関係を確認できない限り、一方の内容で他方を上書きしないでください。
- 十分な根拠がコンテキストにない場合は、その旨を明確に回答してください。

コンテキスト:
{context}

質問:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


# ============================================================
# Metadata helpers
# ============================================================


def document_key(doc):
    return (
        doc.metadata.get("book"),
        doc.metadata.get("pdf_page", doc.metadata.get("page")),
        doc.metadata.get("chunk"),
    )


def page_key_from_doc(doc):
    book = doc.metadata.get("book")
    pdf_page = doc.metadata.get("pdf_page", doc.metadata.get("page"))
    if not book or pdf_page is None:
        return None
    try:
        pdf_page = int(pdf_page)
    except (TypeError, ValueError):
        return None
    return (book, pdf_page)


def get_logical_page(doc) -> int | None:
    value = doc.metadata.get("logical_page")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_pdf_page(doc) -> int | None:
    value = doc.metadata.get("pdf_page", doc.metadata.get("page"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def page_logical_from_index(book: str, pdf_page: int) -> int | None:
    for doc in page_documents_by_pdf.get((book, pdf_page), []):
        logical = get_logical_page(doc)
        if logical is not None:
            return logical
    return None


# ============================================================
# Query normalization / text scoring
# ============================================================


def normalize_search_query(query: str) -> str:
    normalized = query.strip()
    normalized = re.sub(r"[。．.!！?？]+$", "", normalized).strip()

    suffix_patterns = [
        r"について詳しく教えてください$",
        r"について詳しく教えて$",
        r"について教えてください$",
        r"について教えて$",
        r"を詳しく教えてください$",
        r"を詳しく教えて$",
        r"を教えてください$",
        r"を教えて$",
        r"について説明してください$",
        r"について説明して$",
        r"を説明してください$",
        r"を説明して$",
        r"のルールについて$",
        r"のルール$",
        r"とは何ですか$",
        r"とは何$",
        r"って何ですか$",
        r"って何$",
        r"とは$",
    ]
    for pattern in suffix_patterns:
        normalized = re.sub(pattern, "", normalized).strip()
    return normalized or query.strip()


def normalize_navigation_text(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"\s+", "", value)
    value = re.sub(
        r"[《》〈〉「」『』\(\)（）\[\]【】・･/／：:。．,.，!?！？\-―—]",
        "",
        value,
    )
    return value.lower()


def char_ngram_coverage(query: str, text: str, n: int = 2) -> float:
    if not query or not text:
        return 0.0
    if len(query) < n:
        return 1.0 if query in text else 0.0
    qgrams = {query[i : i + n] for i in range(len(query) - n + 1)}
    if not qgrams:
        return 0.0
    return sum(1 for gram in qgrams if gram in text) / len(qgrams)


def extract_reason_terms(reason: str) -> list[str]:
    if not reason:
        return []
    values = []
    for prefix in ("索引: ", "目次: ", "目次セクション続き: "):
        if reason.startswith(prefix):
            value = reason[len(prefix) :].strip()
            if value:
                values.append(value)
    return values


def chunk_relevance_score(
    doc,
    query: str,
    extra_terms: Optional[list[str]] = None,
    prefer_page_start: bool = False,
) -> float:
    text = doc.page_content or ""
    normalized_text = normalize_navigation_text(text)
    normalized_query = normalize_navigation_text(normalize_search_query(query))
    score = 0.0

    if normalized_query:
        if normalized_query in normalized_text:
            score += 25.0
            score += min(8.0, normalized_text.count(normalized_query) * 2.0)
        score += char_ngram_coverage(normalized_query, normalized_text, 2) * 12.0
        if len(normalized_query) >= 4:
            score += char_ngram_coverage(normalized_query, normalized_text, 3) * 8.0

    for term in extra_terms or []:
        normalized_term = normalize_navigation_text(term)
        if not normalized_term:
            continue
        if normalized_term in normalized_text:
            score += 18.0
            score += min(6.0, normalized_text.count(normalized_term) * 1.5)
        else:
            score += char_ngram_coverage(normalized_term, normalized_text, 2) * 6.0

    chunk_no = int(doc.metadata.get("chunk", 0))
    if prefer_page_start and chunk_no == 0:
        score += 5.0

    score *= get_document_authority(doc)
    return score


def build_excerpt(
    doc,
    query: str,
    reason: str = "",
    max_chars: int = CITATION_EXCERPT_CHARS,
) -> str:
    text = re.sub(r"\s+", " ", doc.page_content or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    search_terms = [normalize_search_query(query)]
    search_terms.extend(extract_reason_terms(reason))
    search_terms = [term for term in search_terms if term]
    search_terms.sort(key=len, reverse=True)

    positions = []
    for term in search_terms:
        pos = text.find(term)
        if pos >= 0:
            positions.append((pos, len(term)))

    if positions:
        pos, term_len = min(positions, key=lambda item: item[0])
        center = pos + term_len // 2
        start = max(0, center - max_chars // 2)
        end = min(len(text), start + max_chars)
        if end - start < max_chars:
            start = max(0, end - max_chars)
    else:
        start = 0
        end = min(len(text), max_chars)

    excerpt = text[start:end]
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(text):
        excerpt += "…"
    return excerpt


# ============================================================
# Citation helpers
# ============================================================


def combine_reasons(reasons: list[str]) -> str:
    seen = []
    for reason in reasons:
        if reason and reason not in seen:
            seen.append(reason)
    return " / ".join(seen)


def build_citations(context_items: list[dict], question: str) -> List[Citation]:
    grouped = defaultdict(list)
    page_order = []

    for item in context_items:
        doc = item["doc"]
        key = page_key_from_doc(doc)
        if key is None:
            continue
        if key not in grouped:
            page_order.append(key)
        grouped[key].append(item)

    citations = []
    for key in page_order:
        book, pdf_page = key
        items = grouped[key]
        best_item = max(items, key=lambda item: item.get("context_score", 0.0))
        best_doc = best_item["doc"]
        logical_page = get_logical_page(best_doc)
        if logical_page is None:
            continue

        reasons = [item.get("reason", "") for item in items]
        reason = combine_reasons(reasons)

        citations.append(
            Citation(
                id=len(citations) + 1,
                book=book,
                page=logical_page,
                pdf_page=pdf_page,
                category=get_document_category(best_doc),
                excerpt=build_excerpt(best_doc, question, reason),
                reason=reason or "検索結果から選定",
            )
        )

    return citations


def citations_to_legacy_sources(citations: List[Citation]) -> List[str]:
    result = []
    for citation in citations:
        if citation.category:
            result.append(
                f"{citation.category} / {citation.book} - p.{citation.page}"
            )
        else:
            result.append(f"{citation.book} - p.{citation.page}")
    return result


def extract_used_citation_ids(answer: str) -> list[int]:
    used = []
    for value in re.findall(r"\[C(\d+)\]", answer or ""):
        citation_id = int(value)
        if citation_id not in used:
            used.append(citation_id)
    return used


def filter_used_citations(answer: str, citations: List[Citation]) -> List[Citation]:
    used_ids = set(extract_used_citation_ids(answer))
    if not used_ids:
        return []
    return [citation for citation in citations if citation.id in used_ids]


# ============================================================
# Search
# ============================================================


def definition_search(query: str, top_k: int, books=None):
    term = normalize_search_query(query)
    if not term or len(term) > 30 or " " in term or "　" in term:
        return []

    decorated_term = f"《{term}》"
    definition_markers = ("前提", "適用", "使用", "リスク", "概要", "効果")
    candidates = []

    for doc in search_documents:
        if books and doc.metadata.get("book") not in books:
            continue
        text = doc.page_content
        if term not in text:
            continue

        score = 1.0
        decorated_position = text.find(decorated_term)
        if decorated_position >= 0:
            score += 10.0
            if decorated_position <= 100:
                score += 3.0
            elif decorated_position <= 250:
                score += 2.0
            elif decorated_position <= 500:
                score += 1.0

        score += sum(1 for marker in definition_markers if marker in text) * 0.75
        score *= get_document_authority(doc)
        candidates.append((doc, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:top_k]


def lexical_search(query: str, top_k: int, books=None):
    query_vector = lexical_vectorizer.transform([query])
    scores = cosine_similarity(query_vector, lexical_matrix).flatten()
    sorted_indices = scores.argsort()[::-1]
    results = []

    for index in sorted_indices:
        score = float(scores[index])
        if score <= 0:
            break
        doc = search_documents[index]
        if books and doc.metadata.get("book") not in books:
            continue
        results.append((doc, score))
        if len(results) >= top_k:
            break
    return results


def hybrid_search(query: str, top_k: int, candidate_k: int, books=None):
    search_query = normalize_search_query(query)
    definition_results = definition_search(
        search_query,
        top_k=min(3, top_k),
        books=books,
    )

    vector_results = db.similarity_search_with_score(search_query, k=candidate_k)
    vector_rank = {}
    rank = 0
    for doc, _distance in vector_results:
        if get_logical_page(doc) is None:
            continue
        if books and doc.metadata.get("book") not in books:
            continue
        rank += 1
        vector_rank[document_key(doc)] = (doc, rank)

    lexical_results = lexical_search(search_query, candidate_k, books=books)
    lexical_rank = {
        document_key(doc): (doc, rank)
        for rank, (doc, _score) in enumerate(lexical_results, start=1)
    }

    candidate_keys = set(vector_rank) | set(lexical_rank)
    scored_candidates = []
    rrf_k = 60.0
    vector_weight = 0.40
    lexical_weight = 0.60

    for key in candidate_keys:
        doc = lexical_rank[key][0] if key in lexical_rank else vector_rank[key][0]
        score = 0.0
        reasons = []

        if key in vector_rank:
            vector_position = vector_rank[key][1]
            score += vector_weight / (rrf_k + vector_position)
            reasons.append(f"ベクトル検索 #{vector_position}")

        if key in lexical_rank:
            lexical_position = lexical_rank[key][1]
            score += lexical_weight / (rrf_k + lexical_position)
            reasons.append(f"文字列検索 #{lexical_position}")

        score *= get_document_authority(doc)
        scored_candidates.append((doc, score, reasons))

    scored_candidates.sort(key=lambda item: item[1], reverse=True)

    selected = []
    seen = set()

    for doc, score in definition_results:
        key = document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "doc": doc,
                "retrieval_score": score,
                "reason": "定義候補として直接一致",
            }
        )
        if len(selected) >= top_k:
            return selected

    for doc, score, reasons in scored_candidates:
        key = document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        reason = "通常検索で高関連"
        if reasons:
            reason += "（" + " / ".join(reasons) + "）"
        selected.append(
            {
                "doc": doc,
                "retrieval_score": score,
                "reason": reason,
            }
        )
        if len(selected) >= top_k:
            break

    return selected


# ============================================================
# Navigation search
# ============================================================


def add_navigation_candidate(
    target: dict,
    *,
    book: str,
    pdf_page: int,
    source: str,
    score: float,
    label: str,
    logical_page: int | None = None,
    match_priority: int = 0,
):
    key = (book, pdf_page)
    authority = get_book_authority(book)
    candidate = {
        "book": book,
        "pdf_page": pdf_page,
        "logical_page": logical_page,
        "source": source,
        "score": score,
        "authority": authority,
        "match_priority": match_priority,
        "label": label,
    }
    existing = target.get(key)
    if existing is None:
        target[key] = candidate
        return

    existing_key = (
        existing.get("match_priority", 0),
        existing.get("authority", 1.0),
        existing.get("score", 0.0),
    )
    candidate_key = (match_priority, authority, score)
    if candidate_key > existing_key:
        target[key] = candidate


def navigation_index_search(query: str, books=None):
    search_term = normalize_search_query(query)
    normalized_query = normalize_navigation_text(search_term)
    if not normalized_query:
        return [], False

    raw_matches = []
    exact_match_found = False
    match_priority = {
        "exact": 4,
        "term_in_query": 3,
        "query_in_term": 2,
        "fuzzy": 1,
    }

    for book, navigation in navigation_data.items():
        if books and book not in books:
            continue
        for entry in navigation.get("index", []):
            term = str(entry.get("term", "")).strip()
            if not term:
                continue
            normalized_term = normalize_navigation_text(term)
            if not normalized_term:
                continue

            score = None
            match_type = None
            if normalized_query == normalized_term:
                score = 100.0
                match_type = "exact"
                exact_match_found = True
            elif normalized_term in normalized_query:
                score = 92.0
                match_type = "term_in_query"
            elif normalized_query in normalized_term:
                score = 82.0
                match_type = "query_in_term"
            else:
                ratio = SequenceMatcher(None, normalized_query, normalized_term).ratio()
                if ratio >= NAV_INDEX_FUZZY_THRESHOLD:
                    score = 50.0 + ratio * 40.0
                    match_type = "fuzzy"

            if score is None:
                continue
            raw_matches.append(
                {
                    "book": book,
                    "entry": entry,
                    "term": term,
                    "score": score,
                    "match_type": match_type,
                    "authority": get_book_authority(book),
                }
            )

    raw_matches.sort(
        key=lambda item: (
            match_priority.get(item["match_type"], 0),
            item["authority"],
            item["score"],
            len(item["term"]),
        ),
        reverse=True,
    )

    page_candidates = {}
    for match in raw_matches:
        book = match["book"]
        for pdf_page in match["entry"].get("pdf_pages", []):
            try:
                pdf_page = int(pdf_page)
            except (TypeError, ValueError):
                continue
            add_navigation_candidate(
                page_candidates,
                book=book,
                pdf_page=pdf_page,
                logical_page=page_logical_from_index(book, pdf_page),
                source="index",
                score=match["score"],
                match_priority=match_priority.get(match["match_type"], 0),
                label=f"索引: {match['term']}",
            )

    result = sorted(
        page_candidates.values(),
        key=lambda item: (
            item.get("match_priority", 0),
            item.get("authority", 1.0),
            item["score"],
            -item["pdf_page"],
        ),
        reverse=True,
    )
    return result[:NAV_INDEX_MAX_PAGES], exact_match_found


def navigation_toc_search(query: str, books=None):
    search_term = normalize_search_query(query)
    normalized_query = normalize_navigation_text(search_term)
    if not normalized_query:
        return []

    matches = []
    for book, navigation in navigation_data.items():
        if books and book not in books:
            continue
        for entry in navigation.get("toc", []):
            title = str(entry.get("title", "")).strip()
            if not title:
                continue
            pdf_page = entry.get("pdf_page")
            if pdf_page is None:
                continue
            try:
                pdf_page = int(pdf_page)
            except (TypeError, ValueError):
                continue

            normalized_title = normalize_navigation_text(title)
            if not normalized_title:
                continue

            score = None
            if normalized_query == normalized_title:
                score = 95.0
            elif normalized_title in normalized_query:
                score = 88.0
            elif normalized_query in normalized_title:
                score = 78.0
            else:
                ratio = SequenceMatcher(None, normalized_query, normalized_title).ratio()
                if ratio >= NAV_TOC_FUZZY_THRESHOLD:
                    score = 45.0 + ratio * 35.0

            if score is None:
                continue
            matches.append(
                {
                    "book": book,
                    "pdf_page": pdf_page,
                    "logical_page": entry.get("logical_page"),
                    "source": "toc",
                    "score": score,
                    "authority": get_book_authority(book),
                    "label": f"目次: {title}",
                    "title": title,
                    "level": int(entry.get("level", 1)),
                }
            )

    matches.sort(
        key=lambda item: (
            item["score"],
            item.get("authority", 1.0),
            -item["level"],
        ),
        reverse=True,
    )

    result = []
    seen = set()
    for match in matches:
        key = (match["book"], match["pdf_page"])
        if key in seen:
            continue
        seen.add(key)
        result.append(match)
        if len(result) >= NAV_TOC_MAX_PAGES:
            break
    return result


def build_section_expansion(toc_candidates: list, books=None):
    result = []
    seen = set()
    for toc in toc_candidates:
        book = toc["book"]
        start_pdf = toc["pdf_page"]
        for delta in range(1, NAV_SECTION_EXPAND_PAGES + 1):
            pdf_page = start_pdf + delta
            key = (book, pdf_page)
            if key in seen or key not in page_documents_by_pdf:
                continue
            seen.add(key)
            result.append(
                {
                    "book": book,
                    "pdf_page": pdf_page,
                    "logical_page": page_logical_from_index(book, pdf_page),
                    "source": "section",
                    "score": toc["score"] - 5.0 - delta,
                    "authority": get_book_authority(book),
                    "label": f"目次セクション続き: {toc['title']}",
                    "title": toc["title"],
                    "level": toc.get("level", 1),
                }
            )
            if len(result) >= NAV_SECTION_MAX_PAGES:
                return result
    return result


def navigation_search(query: str, books=None):
    index_candidates, exact_index_match = navigation_index_search(query, books=books)
    toc_candidates = navigation_toc_search(query, books=books)
    section_candidates = []
    if not exact_index_match:
        section_candidates = build_section_expansion(toc_candidates, books=books)

    source_priority = {"index": 3, "toc": 2, "section": 1}
    merged = {}

    for candidate in index_candidates + toc_candidates + section_candidates:
        key = (candidate["book"], candidate["pdf_page"])
        existing = merged.get(key)
        candidate_key = (
            source_priority.get(candidate["source"], 0),
            candidate.get("match_priority", 0),
            candidate.get("authority", 1.0),
            candidate.get("score", 0.0),
        )
        if existing is None:
            merged[key] = candidate
            continue
        existing_key = (
            source_priority.get(existing["source"], 0),
            existing.get("match_priority", 0),
            existing.get("authority", 1.0),
            existing.get("score", 0.0),
        )
        if candidate_key > existing_key:
            merged[key] = candidate

    candidates = list(merged.values())
    candidates.sort(
        key=lambda item: (
            source_priority.get(item["source"], 0),
            item.get("match_priority", 0),
            item.get("authority", 1.0),
            item.get("score", 0.0),
        ),
        reverse=True,
    )
    return candidates[:NAV_MANDATORY_MAX_PAGES], exact_index_match


# ============================================================
# Context reranking / pruning
# ============================================================


def select_navigation_page_documents(candidate: dict, question: str) -> list[dict]:
    docs = list(
        page_documents_by_pdf.get((candidate["book"], candidate["pdf_page"]), [])
    )
    if not docs:
        return []

    prefer_page_start = candidate["source"] in {"toc", "section"}
    terms = extract_reason_terms(candidate.get("label", ""))
    scored = []

    for doc in docs:
        score = chunk_relevance_score(
            doc,
            question,
            extra_terms=terms,
            prefer_page_start=prefer_page_start,
        )
        scored.append((doc, score))

    scored.sort(
        key=lambda item: (
            item[1],
            -int(item[0].metadata.get("chunk", 0)),
        ),
        reverse=True,
    )

    selected = []
    for index, (doc, chunk_score) in enumerate(scored[:NAV_CHUNKS_PER_PAGE]):
        authority_note = ""
        if get_document_authority(doc) > 1.0:
            authority_note = " / 資料優先度を加味"
        selected.append(
            {
                "doc": doc,
                "mandatory": index == 0,
                "context_score": 1000.0 + candidate.get("score", 0.0) + chunk_score,
                "reason": candidate.get("label", "navigationで選定") + authority_note,
                "source": candidate["source"],
            }
        )
    return selected


def build_context_items(
    *,
    question: str,
    navigation_pages: list,
    hybrid_items: list,
) -> list[dict]:
    navigation_items = []
    for candidate in navigation_pages:
        navigation_items.extend(select_navigation_page_documents(candidate, question))

    # navigation各ページの最上位chunkは必須。
    mandatory_items = [item for item in navigation_items if item["mandatory"]]
    optional_navigation = [item for item in navigation_items if not item["mandatory"]]

    hybrid_context_items = []
    for rank, item in enumerate(hybrid_items, start=1):
        doc = item["doc"]
        relevance = chunk_relevance_score(doc, question)
        # retrieval_scoreはRRF等で桁が小さいため、順位を主な信号にする。
        rank_bonus = max(0.0, 30.0 - rank)
        authority_note = ""
        if get_document_authority(doc) > 1.0:
            authority_note = " / 資料優先度を加味"
        hybrid_context_items.append(
            {
                "doc": doc,
                "mandatory": False,
                "context_score": 100.0 + rank_bonus + relevance,
                "reason": item.get("reason", "通常検索で高関連") + authority_note,
                "source": "hybrid",
            }
        )

    mandatory_items.sort(key=lambda item: item["context_score"], reverse=True)
    optional_pool = optional_navigation + hybrid_context_items
    optional_pool.sort(key=lambda item: item["context_score"], reverse=True)

    selected = []
    seen = set()

    def add_item(item: dict) -> bool:
        key = document_key(item["doc"])
        if key in seen:
            return False
        seen.add(key)
        selected.append(item)
        return True

    # navigation必須ページは上限内で無条件採用。
    for item in mandatory_items:
        if len(selected) >= CONTEXT_MAX_DOCS:
            break
        add_item(item)

    # 余った枠を、追加navigation chunkとHybrid候補の再ランキングで埋める。
    for item in optional_pool:
        if len(selected) >= CONTEXT_MAX_DOCS:
            break
        add_item(item)

    return selected


# ============================================================
# Exact-search helpers
# ============================================================


def apply_category_weight(results):
    return sorted(results, key=get_document_authority, reverse=True)


def build_exact_search_items(results, question: str) -> list[dict]:
    # 同一ページのchunkをまとめ、質問に最も近いchunkを代表にする。
    grouped = defaultdict(list)
    page_order = []
    for doc in results:
        key = page_key_from_doc(doc)
        if key is None:
            continue
        if key not in grouped:
            page_order.append(key)
        grouped[key].append(doc)

    items = []
    for key in page_order:
        docs = grouped[key]
        best_doc = max(docs, key=lambda doc: chunk_relevance_score(doc, question))
        items.append(
            {
                "doc": best_doc,
                "mandatory": True,
                "context_score": chunk_relevance_score(best_doc, question),
                "reason": f"全文検索: 「{question.strip()}」に一致",
                "source": "exact",
            }
        )
    return items


# ============================================================
# /ask
# ============================================================


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    question = request.question
    books = request.books
    model_name = request.model or "gpt-5.4-nano"
    mode = request.mode or "rules_strict"
    initial_k = max(1, int(request.k or 10))
    max_k = HYBRID_CANDIDATE_K

    if mode == "free_chat":
        system_prompt = (
            "あなたはソード・ワールド2.5の世界観とルールに精通したAIです。"
            "ユーザーの質問には必ずSW2.5の文脈で、具体的かつ専門的に回答してください。"
        )
        llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ]
        )
        return QueryResponse(
            answer=response.content,
            citations=[],
            sources=[],
            model_used=model_name,
            k_used=0,
            hybrid_k_used=0,
            navigation_pages_used=0,
            max_k=0,
            token_usage=response.response_metadata.get("token_usage", {}),
        )

    if mode == "exact_search":
        keywords = question.strip().split()
        results = []
        for doc in search_documents:
            if books and doc.metadata.get("book") not in books:
                continue
            if all(keyword in doc.page_content for keyword in keywords):
                results.append(doc)

        results = apply_category_weight(results)
        exact_items = build_exact_search_items(results, question)
        citations = build_citations(exact_items, question)
        sources = citations_to_legacy_sources(citations)
        answer = (
            "全文検索を実施しました。結果は出典に記載されています。"
            if sources
            else "該当はありませんでした。"
        )
        return QueryResponse(
            answer=answer,
            citations=citations,
            sources=sources,
            model_used="AIは使用していません",
            k_used=0,
            hybrid_k_used=0,
            navigation_pages_used=0,
            max_k=0,
            token_usage={},
        )

    # rules_strict
    hybrid_items = hybrid_search(
        query=question,
        top_k=initial_k,
        candidate_k=max_k,
        books=books,
    )
    navigation_pages, _exact_index_match = navigation_search(
        query=question,
        books=books,
    )

    context_items = build_context_items(
        question=question,
        navigation_pages=navigation_pages,
        hybrid_items=hybrid_items,
    )

    if not context_items:
        return QueryResponse(
            answer="該当する情報が見つかりませんでした。",
            citations=[],
            sources=[],
            model_used=model_name,
            k_used=0,
            hybrid_k_used=0,
            navigation_pages_used=0,
            max_k=max_k,
            token_usage={},
        )

    citations = build_citations(context_items, question)
    citation_id_map = {
        (citation.book, citation.pdf_page): citation.id for citation in citations
    }

    context_parts = []
    for item in context_items:
        doc = item["doc"]
        book = doc.metadata.get("book", "不明")
        logical_page = get_logical_page(doc)
        pdf_page = get_pdf_page(doc)
        if logical_page is None or pdf_page is None:
            continue

        citation_id = citation_id_map.get((book, pdf_page))
        if citation_id is None:
            continue

        reason = item.get("reason", "")
        reason_text = f"\n検索補助情報: {reason}" if reason else ""
        context_parts.append(
            f"[CITATION:C{citation_id}]\n"
            f"書籍: {book}\n"
            f"書籍ページ: {logical_page}"
            f"{reason_text}\n"
            f"本文:\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)
    full_prompt = prompt.format(context=context, question=question)

    llm = ChatOpenAI(model_name=model_name, temperature=0)
    response = llm.invoke(full_prompt)

    # UIの出典一覧には、LLMが実際に[Cx]で使用したページだけ返す。
    used_citations = filter_used_citations(response.content, citations)
    sources = citations_to_legacy_sources(used_citations)

    return QueryResponse(
        answer=response.content,
        citations=used_citations,
        sources=sources,
        model_used=model_name,
        k_used=len(context_items),
        hybrid_k_used=len(hybrid_items),
        navigation_pages_used=len(navigation_pages),
        max_k=max_k,
        token_usage=response.response_metadata.get("token_usage", {}),
    )
