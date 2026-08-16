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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage


# ============================================================
# Environment
# ============================================================

load_dotenv()


INDEX_DIR = os.getenv(
    "INDEX_DIR",
    "./vector_index",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

BOOK_CATEGORY_PATH = Path(
    os.getenv(
        "BOOK_CATEGORY_PATH",
        "./metadata/book_categories.json",
    )
)

NAVIGATION_DIR = Path(
    os.getenv(
        "NAVIGATION_DIR",
        "./metadata/navigation",
    )
)


# ============================================================
# Hybrid search settings
# ============================================================

HYBRID_CANDIDATE_K = int(
    os.getenv(
        "HYBRID_CANDIDATE_K",
        "100",
    )
)


# ============================================================
# Source authority settings
# ============================================================

# book_categories.json の weight を、
# 検索順位へどの程度反映するか。
#
# 例:
#   weight=1.0 -> x1.00
#   weight=2.0 -> x1.20
#   weight=3.0 -> x1.35（上限）
SOURCE_AUTHORITY_STRENGTH = float(
    os.getenv(
        "SOURCE_AUTHORITY_STRENGTH",
        "0.20",
    )
)

SOURCE_AUTHORITY_MAX = float(
    os.getenv(
        "SOURCE_AUTHORITY_MAX",
        "1.35",
    )
)


# ============================================================
# Navigation settings
# ============================================================

# 索引由来で強制追加する最大ページ数
NAV_INDEX_MAX_PAGES = int(
    os.getenv(
        "NAV_INDEX_MAX_PAGES",
        "8",
    )
)

# 目次検索そのものから強制追加する最大ページ数
NAV_TOC_MAX_PAGES = int(
    os.getenv(
        "NAV_TOC_MAX_PAGES",
        "4",
    )
)

# 広いセクション検索時、
# 目次開始ページから何ページ先まで補完するか。
NAV_SECTION_EXPAND_PAGES = int(
    os.getenv(
        "NAV_SECTION_EXPAND_PAGES",
        "2",
    )
)

# section expansion全体の最大ページ数
NAV_SECTION_MAX_PAGES = int(
    os.getenv(
        "NAV_SECTION_MAX_PAGES",
        "6",
    )
)

# navigation由来ページ全体の絶対上限
NAV_MANDATORY_MAX_PAGES = int(
    os.getenv(
        "NAV_MANDATORY_MAX_PAGES",
        "12",
    )
)

# navigationページ1枚からLLMへ渡す最大chunk数
NAV_CHUNKS_PER_PAGE = int(
    os.getenv(
        "NAV_CHUNKS_PER_PAGE",
        "3",
    )
)

# 最終的にLLMへ渡すchunk全体の最大数。
#
# navigationページはkとは別枠だが、
# context自体は無制限にしない。
CONTEXT_MAX_DOCS = int(
    os.getenv(
        "CONTEXT_MAX_DOCS",
        "48",
    )
)

NAV_INDEX_FUZZY_THRESHOLD = float(
    os.getenv(
        "NAV_INDEX_FUZZY_THRESHOLD",
        "0.72",
    )
)

NAV_TOC_FUZZY_THRESHOLD = float(
    os.getenv(
        "NAV_TOC_FUZZY_THRESHOLD",
        "0.60",
    )
)

NAVIGATION_REQUIRED = (
    os.getenv(
        "NAVIGATION_REQUIRED",
        "1",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


# ============================================================
# FAISS
# ============================================================

embedding = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)

db = FAISS.load_local(
    INDEX_DIR,
    embedding,
    allow_dangerous_deserialization=True,
)

all_index_documents = list(
    db.docstore._dict.values()
)


# ============================================================
# Search documents
# ============================================================

# 書籍ページ番号を持たない表紙・カバー等は
# strict rule retrievalから除外する。
search_documents = [
    doc
    for doc in all_index_documents
    if doc.metadata.get(
        "logical_page"
    ) is not None
]


# ============================================================
# Page -> chunks lookup
# ============================================================

page_documents_by_pdf = defaultdict(
    list
)

for doc in search_documents:
    book = doc.metadata.get(
        "book"
    )

    pdf_page = doc.metadata.get(
        "pdf_page",
        doc.metadata.get(
            "page"
        ),
    )

    if not book or pdf_page is None:
        continue

    try:
        pdf_page = int(
            pdf_page
        )
    except (
        TypeError,
        ValueError,
    ):
        continue

    page_documents_by_pdf[
        (
            book,
            pdf_page,
        )
    ].append(
        doc
    )


for docs in page_documents_by_pdf.values():
    docs.sort(
        key=lambda doc: int(
            doc.metadata.get(
                "chunk",
                0,
            )
        )
    )


# ============================================================
# Lexical search
# ============================================================

lexical_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    min_df=1,
)

lexical_matrix = (
    lexical_vectorizer.fit_transform(
        [
            doc.page_content
            for doc in search_documents
        ]
    )
)


# ============================================================
# Book categories
# ============================================================

with BOOK_CATEGORY_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    category_data = json.load(
        f
    )


book_to_category = {}
category_weight = {}


sorted_categories = sorted(
    category_data.items(),
    key=lambda item: item[1].get(
        "weight",
        1.0,
    ),
    reverse=True,
)


for category, info in sorted_categories:

    books_sorted = sorted(
        info["books"],
        key=lambda x: x["name"],
    )

    for book_entry in books_sorted:

        book_name = book_entry[
            "name"
        ]

        book_to_category[
            book_name
        ] = category

    category_weight[
        category
    ] = info.get(
        "weight",
        1.0,
    )


# ============================================================
# Navigation loading
# ============================================================

def load_navigation_data(
    path: Path,
) -> dict:
    if not path.exists():

        if NAVIGATION_REQUIRED:
            raise FileNotFoundError(
                "navigation directory "
                f"not found: {path}"
            )

        return {}

    json_files = sorted(
        path.glob(
            "*.json"
        )
    )

    if (
        not json_files
        and NAVIGATION_REQUIRED
    ):
        raise RuntimeError(
            "navigation JSONがありません: "
            f"{path}"
        )

    result = {}

    for json_path in json_files:

        try:
            with json_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(
                    f
                )

        except Exception as exc:
            print(
                "WARNING: navigation JSON "
                f"load failed: {json_path}: "
                f"{exc}"
            )

            continue

        if not isinstance(
            data,
            dict,
        ):
            continue

        book = data.get(
            "book"
        )

        if not book:
            continue

        result[
            book
        ] = data

    return result


navigation_data = (
    load_navigation_data(
        NAVIGATION_DIR
    )
)

print(
    "Navigation books loaded: "
    f"{len(navigation_data)}"
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI()


# ============================================================
# API models
# ============================================================

class QueryRequest(BaseModel):
    question: str
    books: Optional[List[str]] = None
    model: Optional[str] = (
        "gpt-5.4-nano"
    )
    k: Optional[int] = 10
    mode: Optional[str] = (
        "rules_strict"
    )


class Citation(BaseModel):
    id: int

    book: str

    # 実際の書籍に印刷されているページ
    page: int

    # PDF / JPEG内部ページ
    pdf_page: int

    category: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str

    citations: List[
        Citation
    ] = Field(
        default_factory=list
    )

    # 旧クライアント互換
    sources: List[str] = Field(
        default_factory=list
    )

    model_used: Optional[str] = None

    # 最終contextに入ったchunk数
    k_used: Optional[int] = None

    # 通常Hybrid Searchへ指定したk
    hybrid_k_used: Optional[int] = None

    # navigationから強制追加したページ数
    navigation_pages_used: Optional[int] = None

    max_k: Optional[int] = None

    token_usage: Optional[dict] = None


# ============================================================
# Prompt
# ============================================================

template = """
以下のコンテキストだけを根拠として、
質問に正確に答えてください。

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
- 基本資料と追加資料で説明内容が異なる場合、コンテキストから変更・追加関係を確認できない限り、一方の内容で他方を上書きしないでください。- 十分な根拠がコンテキストにない場合は、その旨を明確に回答してください。

コンテキスト:
{context}

質問:
{question}
"""


prompt = PromptTemplate(
    input_variables=[
        "context",
        "question",
    ],
    template=template,
)


# ============================================================
# Metadata helpers
# ============================================================

def document_key(
    doc,
):
    return (
        doc.metadata.get(
            "book"
        ),
        doc.metadata.get(
            "pdf_page",
            doc.metadata.get(
                "page"
            ),
        ),
        doc.metadata.get(
            "chunk"
        ),
    )


def page_key_from_doc(
    doc,
):
    book = doc.metadata.get(
        "book"
    )

    pdf_page = doc.metadata.get(
        "pdf_page",
        doc.metadata.get(
            "page"
        ),
    )

    if (
        not book
        or pdf_page is None
    ):
        return None

    try:
        pdf_page = int(
            pdf_page
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    return (
        book,
        pdf_page,
    )


def get_document_category(
    doc,
):
    book = doc.metadata.get(
        "book"
    )

    return doc.metadata.get(
        "category",
        book_to_category.get(
            book,
            "その他",
        ),
    )


def get_book_authority(
    book: str,
) -> float:
    """
    書籍カテゴリのweightを検索用authorityへ変換する。

    生のweightをそのまま乗算すると影響が強すぎるため、
    1.0～SOURCE_AUTHORITY_MAXへ圧縮する。
    """

    category = book_to_category.get(
        book,
        "その他",
    )

    raw_weight = float(
        category_weight.get(
            category,
            1.0,
        )
    )

    authority = (
        1.0
        + max(
            0.0,
            raw_weight - 1.0,
        )
        * SOURCE_AUTHORITY_STRENGTH
    )

    return min(
        SOURCE_AUTHORITY_MAX,
        authority,
    )


def get_document_authority(
    doc,
) -> float:
    book = doc.metadata.get(
        "book",
        "",
    )

    return get_book_authority(
        book
    )


def get_logical_page(
    doc,
) -> int | None:
    value = doc.metadata.get(
        "logical_page"
    )

    if value is None:
        return None

    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def get_pdf_page(
    doc,
) -> int | None:
    value = doc.metadata.get(
        "pdf_page",
        doc.metadata.get(
            "page"
        ),
    )

    if value is None:
        return None

    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# Citations
# ============================================================

def build_citations(
    documents,
) -> List[Citation]:
    """
    表示ページはlogical_page。
    リンク用にはpdf_pageを保持する。
    """

    citations = []
    seen = set()

    for doc in documents:

        book = doc.metadata.get(
            "book"
        )

        logical_page = (
            get_logical_page(
                doc
            )
        )

        pdf_page = (
            get_pdf_page(
                doc
            )
        )

        if (
            not book
            or logical_page is None
            or pdf_page is None
        ):
            continue

        key = (
            book,
            pdf_page,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        citations.append(
            Citation(
                id=(
                    len(citations)
                    + 1
                ),
                book=book,
                page=(
                    logical_page
                ),
                pdf_page=(
                    pdf_page
                ),
                category=(
                    get_document_category(
                        doc
                    )
                ),
            )
        )

    return citations


def citations_to_legacy_sources(
    citations: List[
        Citation
    ],
) -> List[str]:

    sources = []

    for citation in citations:

        if citation.category:

            sources.append(
                f"{citation.category} / "
                f"{citation.book} "
                f"- p.{citation.page}"
            )

        else:

            sources.append(
                f"{citation.book} "
                f"- p.{citation.page}"
            )

    return sources


# ============================================================
# Category weight
# ============================================================

def apply_category_weight(
    results,
):

    boosted_results = []

    for doc in results:

        category = (
            get_document_category(
                doc
            )
        )

        weight = (
            category_weight.get(
                category,
                1.0,
            )
        )

        boosted_results.append(
            (
                doc,
                weight,
            )
        )

    boosted_results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        doc
        for doc, _
        in boosted_results
    ]


# ============================================================
# Query normalization
# ============================================================

def normalize_search_query(
    query: str,
) -> str:

    normalized = (
        query.strip()
    )

    normalized = re.sub(
        r"[。．.!！?？]+$",
        "",
        normalized,
    ).strip()

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

        normalized = re.sub(
            pattern,
            "",
            normalized,
        ).strip()

    if not normalized:
        return query.strip()

    return normalized


def normalize_navigation_text(
    value: str,
) -> str:
    """
    索引/目次名比較用。

    空白・装飾記号等を除去するが、
    日本語文字そのものは保持する。
    """

    value = str(
        value
        or ""
    )

    value = value.strip()

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    value = re.sub(
        r"[《》〈〉「」『』"
        r"\(\)（）"
        r"\[\]【】"
        r"・･/／"
        r"：:"
        r"。．,.，"
        r"!?！？"
        r"\-―—]",
        "",
        value,
    )

    return value.lower()


# ============================================================
# Definition search
# ============================================================

def definition_search(
    query: str,
    top_k: int,
    books=None,
):

    term = normalize_search_query(
        query
    )

    if (
        not term
        or len(term) > 30
        or " " in term
        or "　" in term
    ):
        return []

    decorated_term = (
        f"《{term}》"
    )

    definition_markers = (
        "前提",
        "適用",
        "使用",
        "リスク",
        "概要",
        "効果",
    )

    candidates = []

    for doc in search_documents:

        if (
            books
            and doc.metadata.get(
                "book"
            )
            not in books
        ):
            continue

        text = doc.page_content

        if term not in text:
            continue

        score = 0.0

        decorated_position = (
            text.find(
                decorated_term
            )
        )

        if decorated_position >= 0:

            score += 10.0

            if decorated_position <= 100:
                score += 3.0

            elif decorated_position <= 250:
                score += 2.0

            elif decorated_position <= 500:
                score += 1.0

        marker_count = sum(
            1
            for marker
            in definition_markers
            if marker in text
        )

        score += (
            marker_count
            * 0.75
        )

        # 単純な語出現にも最低点を与える
        score += 1.0

        score *= get_document_authority(
            doc
        )

        candidates.append(
            (
                doc,
                score,
            )
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        doc
        for doc, _score
        in candidates[
            :top_k
        ]
    ]


# ============================================================
# Lexical search
# ============================================================

def lexical_search(
    query: str,
    top_k: int,
    books=None,
):

    query_vector = (
        lexical_vectorizer
        .transform(
            [query]
        )
    )

    scores = cosine_similarity(
        query_vector,
        lexical_matrix,
    ).flatten()

    sorted_indices = (
        scores
        .argsort()[::-1]
    )

    results = []

    for index in sorted_indices:

        score = float(
            scores[
                index
            ]
        )

        if score <= 0:
            break

        doc = search_documents[
            index
        ]

        if (
            books
            and doc.metadata.get(
                "book"
            )
            not in books
        ):
            continue

        results.append(
            (
                doc,
                score,
            )
        )

        if len(
            results
        ) >= top_k:
            break

    return results


# ============================================================
# Hybrid search
# ============================================================

def hybrid_search(
    query: str,
    top_k: int,
    candidate_k: int,
    books=None,
):

    search_query = (
        normalize_search_query(
            query
        )
    )

    definition_results = (
        definition_search(
            query=(
                search_query
            ),
            top_k=min(
                3,
                top_k,
            ),
            books=books,
        )
    )

    vector_results = (
        db.similarity_search_with_score(
            search_query,
            k=candidate_k,
        )
    )

    vector_rank = {}

    rank = 0

    for doc, _distance in (
        vector_results
    ):

        # logical pageなしは除外
        if get_logical_page(
            doc
        ) is None:
            continue

        if (
            books
            and doc.metadata.get(
                "book"
            )
            not in books
        ):
            continue

        rank += 1

        vector_rank[
            document_key(
                doc
            )
        ] = (
            doc,
            rank,
        )

    lexical_results = (
        lexical_search(
            query=(
                search_query
            ),
            top_k=(
                candidate_k
            ),
            books=books,
        )
    )

    lexical_rank = {}

    for rank, (
        doc,
        _score,
    ) in enumerate(
        lexical_results,
        start=1,
    ):

        lexical_rank[
            document_key(
                doc
            )
        ] = (
            doc,
            rank,
        )

    candidate_keys = (
        set(
            vector_rank.keys()
        )
        |
        set(
            lexical_rank.keys()
        )
    )

    scored_candidates = []

    rrf_k = 60.0

    vector_weight = 0.40
    lexical_weight = 0.60

    for key in candidate_keys:

        if key in lexical_rank:

            doc = lexical_rank[
                key
            ][0]

        else:

            doc = vector_rank[
                key
            ][0]

        score = 0.0

        if key in vector_rank:

            rank = vector_rank[
                key
            ][1]

            score += (
                vector_weight
                /
                (
                    rrf_k
                    + rank
                )
            )

        if key in lexical_rank:

            rank = lexical_rank[
                key
            ][1]

            score += (
                lexical_weight
                /
                (
                    rrf_k
                    + rank
                )
            )

        score *= get_document_authority(
            doc
        )

        scored_candidates.append(
            (
                doc,
                score,
            )
        )

    scored_candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    selected_docs = []
    seen_keys = set()

    for doc in definition_results:

        key = document_key(
            doc
        )

        if key in seen_keys:
            continue

        seen_keys.add(
            key
        )

        selected_docs.append(
            doc
        )

        if len(
            selected_docs
        ) >= top_k:
            return selected_docs

    for doc, _score in (
        scored_candidates
    ):

        key = document_key(
            doc
        )

        if key in seen_keys:
            continue

        seen_keys.add(
            key
        )

        selected_docs.append(
            doc
        )

        if len(
            selected_docs
        ) >= top_k:
            break

    return selected_docs


# ============================================================
# Navigation candidate helpers
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
    key = (
        book,
        pdf_page,
    )

    authority = (
        get_book_authority(
            book
        )
    )

    existing = target.get(
        key
    )

    candidate = {
        "book": book,
        "pdf_page": pdf_page,
        "logical_page": (
            logical_page
        ),
        "source": source,
        "score": score,
        "authority": authority,
        "match_priority": (
            match_priority
        ),
        "label": label,
    }

    if existing is None:

        target[
            key
        ] = candidate

        return

    existing_key = (
        existing.get(
            "match_priority",
            0,
        ),
        existing.get(
            "authority",
            1.0,
        ),
        existing.get(
            "score",
            0.0,
        ),
    )

    candidate_key = (
        candidate[
            "match_priority"
        ],
        candidate[
            "authority"
        ],
        candidate[
            "score"
        ],
    )

    if candidate_key > existing_key:
        target[
            key
        ] = candidate


def page_logical_from_index(
    book: str,
    pdf_page: int,
) -> int | None:
    docs = page_documents_by_pdf.get(
        (
            book,
            pdf_page,
        ),
        [],
    )

    for doc in docs:

        logical = (
            get_logical_page(
                doc
            )
        )

        if logical is not None:
            return logical

    return None


# ============================================================
# Navigation INDEX search
# ============================================================

def navigation_index_search(
    query: str,
    books=None,
):
    """
    索引項目を検索する。

    exact > contains > fuzzy
    の順に評価する。
    """

    search_term = (
        normalize_search_query(
            query
        )
    )

    normalized_query = (
        normalize_navigation_text(
            search_term
        )
    )

    if not normalized_query:
        return (
            [],
            False,
        )

    raw_matches = []

    exact_match_found = False

    for book, navigation in (
        navigation_data.items()
    ):

        if (
            books
            and book not in books
        ):
            continue

        for entry in navigation.get(
            "index",
            [],
        ):

            term = str(
                entry.get(
                    "term",
                    "",
                )
            ).strip()

            if not term:
                continue

            normalized_term = (
                normalize_navigation_text(
                    term
                )
            )

            if not normalized_term:
                continue

            score = None
            match_type = None

            if (
                normalized_query
                == normalized_term
            ):
                score = 100.0
                match_type = "exact"
                exact_match_found = True

            elif (
                normalized_term
                in normalized_query
            ):
                score = 92.0
                match_type = "term_in_query"

            elif (
                normalized_query
                in normalized_term
            ):
                score = 82.0
                match_type = "query_in_term"

            else:

                ratio = (
                    SequenceMatcher(
                        None,
                        normalized_query,
                        normalized_term,
                    ).ratio()
                )

                if (
                    ratio
                    >= NAV_INDEX_FUZZY_THRESHOLD
                ):
                    score = (
                        50.0
                        + ratio
                        * 40.0
                    )

                    match_type = (
                        "fuzzy"
                    )

            if score is None:
                continue

            raw_matches.append(
                {
                    "book": book,
                    "entry": entry,
                    "term": term,
                    "score": score,
                    "match_type": (
                        match_type
                    ),
                    "authority": (
                        get_book_authority(
                            book
                        )
                    ),
                }
            )

    match_priority = {
        "exact": 4,
        "term_in_query": 3,
        "query_in_term": 2,
        "fuzzy": 1,
    }

    raw_matches.sort(
        key=lambda item: (
            match_priority.get(
                item[
                    "match_type"
                ],
                0,
            ),

            # 同じ一致種別なら、
            # 基本資料等を優先
            item[
                "authority"
            ],

            # fuzzy同士などでは元の関連度も利用
            item[
                "score"
            ],

            # より具体的な項目名を若干優先
            len(
                item[
                    "term"
                ]
            ),
        ),
        reverse=True,
    )

    page_candidates = {}

    for match in raw_matches:

        book = match[
            "book"
        ]

        entry = match[
            "entry"
        ]

        pdf_pages = (
            entry.get(
                "pdf_pages",
                [],
            )
        )

        for pdf_page in pdf_pages:

            try:
                pdf_page = int(
                    pdf_page
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            logical_page = (
                page_logical_from_index(
                    book,
                    pdf_page,
                )
            )

            add_navigation_candidate(
                page_candidates,
                book=book,
                pdf_page=(
                    pdf_page
                ),
                logical_page=(
                    logical_page
                ),
                source="index",
                score=(
                    match[
                        "score"
                    ]
                ),
                match_priority=(
                    match_priority.get(
                        match[
                            "match_type"
                        ],
                        0,
                    )
                ),
                label=(
                    f"索引: "
                    f"{match['term']}"
                ),
            )

    result = sorted(
        page_candidates.values(),
        key=lambda item: (
            item.get(
                "match_priority",
                0,
            ),
            item.get(
                "authority",
                1.0,
            ),
            item[
                "score"
            ],
            -item[
                "pdf_page"
            ],
        ),
        reverse=True,
    )

    return (
        result[
            :NAV_INDEX_MAX_PAGES
        ],
        exact_match_found,
    )


# ============================================================
# Navigation TOC search
# ============================================================

def navigation_toc_search(
    query: str,
    books=None,
):
    search_term = (
        normalize_search_query(
            query
        )
    )

    normalized_query = (
        normalize_navigation_text(
            search_term
        )
    )

    if not normalized_query:
        return []

    matches = []

    for book, navigation in (
        navigation_data.items()
    ):

        if (
            books
            and book not in books
        ):
            continue

        for entry in navigation.get(
            "toc",
            [],
        ):

            title = str(
                entry.get(
                    "title",
                    "",
                )
            ).strip()

            if not title:
                continue

            pdf_page = entry.get(
                "pdf_page"
            )

            if pdf_page is None:
                continue

            try:
                pdf_page = int(
                    pdf_page
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            normalized_title = (
                normalize_navigation_text(
                    title
                )
            )

            if not normalized_title:
                continue

            score = None

            if (
                normalized_query
                == normalized_title
            ):
                score = 95.0

            elif (
                normalized_title
                in normalized_query
            ):
                score = 88.0

            elif (
                normalized_query
                in normalized_title
            ):
                score = 78.0

            else:

                ratio = (
                    SequenceMatcher(
                        None,
                        normalized_query,
                        normalized_title,
                    ).ratio()
                )

                if (
                    ratio
                    >= NAV_TOC_FUZZY_THRESHOLD
                ):
                    score = (
                        45.0
                        + ratio
                        * 35.0
                    )

            if score is None:
                continue

            matches.append(
                {
                    "book": book,
                    "pdf_page": (
                        pdf_page
                    ),
                    "logical_page": (
                        entry.get(
                            "logical_page"
                        )
                    ),
                    "source": "toc",
                    "score": score,

                    # 追加
                    "authority": (
                        get_book_authority(
                            book
                        )
                    ),

                    "label": (
                        f"目次: {title}"
                    ),
                    "title": title,
                    "level": int(
                        entry.get(
                            "level",
                            1,
                        )
                    ),
                }
            )

    matches.sort(
        key=lambda item: (
            item[
                "score"
            ],
            item.get(
                "authority",
                1.0,
            ),
            -item[
                "level"
            ],
        ),
        reverse=True,
    )

    result = []
    seen = set()

    for match in matches:

        key = (
            match["book"],
            match["pdf_page"],
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            match
        )

        if len(
            result
        ) >= NAV_TOC_MAX_PAGES:
            break

    return result


# ============================================================
# Section expansion
# ============================================================

def build_section_expansion(
    toc_candidates: list,
    books=None,
):
    """
    広いルール・章検索向け。

    目次の開始ページだけでなく、
    その直後の数ページもnavigation枠へ追加する。

    固有項目exact index matchの場合は呼ばない。
    """

    result = []
    seen = set()

    for toc in toc_candidates:

        book = toc[
            "book"
        ]

        start_pdf = toc[
            "pdf_page"
        ]

        for delta in range(
            1,
            NAV_SECTION_EXPAND_PAGES
            + 1,
        ):
            pdf_page = (
                start_pdf
                + delta
            )

            key = (
                book,
                pdf_page,
            )

            if key in seen:
                continue

            if key not in (
                page_documents_by_pdf
            ):
                continue

            seen.add(
                key
            )

            result.append(
                {
                    "book": book,
                    "pdf_page": (
                        pdf_page
                    ),
                    "logical_page": (
                        page_logical_from_index(
                            book,
                            pdf_page,
                        )
                    ),
                    "source": (
                        "section"
                    ),
                    "score": (
                        toc["score"]
                        - 5.0
                        - delta
                    ),
                    "label": (
                        f"目次セクション続き: "
                        f"{toc['title']}"
                    ),
                    "authority": (
                        get_book_authority(
                            book
                        )
                    ),
                }
            )

            if len(
                result
            ) >= NAV_SECTION_MAX_PAGES:
                return result

    return result


# ============================================================
# Build mandatory navigation pages
# ============================================================

def navigation_search(
    query: str,
    books=None,
):
    index_candidates, (
        exact_index_match
    ) = navigation_index_search(
        query=query,
        books=books,
    )

    toc_candidates = (
        navigation_toc_search(
            query=query,
            books=books,
        )
    )

    # 索引exact matchがある場合、
    # 「魔法」「アイテム」「種族名」などの
    # 固有項目検索とみなし、
    # section expansionは行わない。
    section_candidates = []

    if not exact_index_match:
        section_candidates = (
            build_section_expansion(
                toc_candidates,
                books=books,
            )
        )

    merged = {}

    # 優先順位:
    # index -> toc -> section
    for candidate in (
        index_candidates
        + toc_candidates
        + section_candidates
    ):

        key = (
            candidate[
                "book"
            ],
            candidate[
                "pdf_page"
            ],
        )

        existing = merged.get(
            key
        )

        if (
            existing is None
            or candidate[
                "score"
            ]
            > existing[
                "score"
            ]
        ):
            merged[
                key
            ] = candidate

    candidates = list(
        merged.values()
    )

    source_priority = {
        "index": 3,
        "toc": 2,
        "section": 1,
    }

    candidates.sort(
        key=lambda item: (
            source_priority.get(
                item[
                    "source"
                ],
                0,
            ),

            item.get(
                "match_priority",
                0,
            ),

            item.get(
                "authority",
                get_book_authority(
                    item[
                        "book"
                    ]
                ),
            ),

            item[
                "score"
            ],
        ),
        reverse=True,
    )

    return (
        candidates[
            :NAV_MANDATORY_MAX_PAGES
        ],
        exact_index_match,
    )


# ============================================================
# Select chunks from one page
# ============================================================

def select_page_documents(
    *,
    book: str,
    pdf_page: int,
    query: str,
    max_chunks: int,
):
    docs = list(
        page_documents_by_pdf.get(
            (
                book,
                pdf_page,
            ),
            [],
        )
    )

    if not docs:
        return []

    if len(
        docs
    ) <= max_chunks:
        return docs

    search_term = (
        normalize_search_query(
            query
        )
    )

    normalized_term = (
        normalize_navigation_text(
            search_term
        )
    )

    scored = []

    for doc in docs:

        text = doc.page_content

        normalized_text = (
            normalize_navigation_text(
                text
            )
        )

        score = 0.0

        chunk_no = int(
            doc.metadata.get(
                "chunk",
                0,
            )
        )

        # ページ先頭chunkは、
        # section冒頭・見出しを保持しやすい。
        if chunk_no == 0:
            score += 3.0

        if (
            normalized_term
            and normalized_term
            in normalized_text
        ):
            score += 10.0

            score += min(
                5.0,
                normalized_text.count(
                    normalized_term
                ),
            )

        scored.append(
            (
                doc,
                score,
                chunk_no,
            )
        )

    scored.sort(
        key=lambda item: (
            item[1],
            -item[2],
        ),
        reverse=True,
    )

    selected = [
        doc
        for doc, _score, _chunk
        in scored[
            :max_chunks
        ]
    ]

    # 最終的にはページ順へ戻す。
    selected.sort(
        key=lambda doc: int(
            doc.metadata.get(
                "chunk",
                0,
            )
        )
    )

    return selected


# ============================================================
# Merge navigation + hybrid context
# ============================================================

def build_context_documents(
    *,
    question: str,
    navigation_pages: list,
    hybrid_documents: list,
):
    """
    navigationはkとは別枠。

    ただしCONTEXT_MAX_DOCSは超えない。

    navigationページについては、
    まず最低1chunkずつ確実に入れ、
    その後追加chunk、
    最後に通常Hybrid結果を追加する。
    """

    selected = []
    seen_document_keys = set()

    navigation_doc_groups = []

    # --------------------------------------------------------
    # NavigationページをDocument化
    # --------------------------------------------------------

    for candidate in navigation_pages:

        docs = select_page_documents(
            book=(
                candidate[
                    "book"
                ]
            ),
            pdf_page=(
                candidate[
                    "pdf_page"
                ]
            ),
            query=(
                question
            ),
            max_chunks=(
                NAV_CHUNKS_PER_PAGE
            ),
        )

        if not docs:
            continue

        navigation_doc_groups.append(
            (
                candidate,
                docs,
            )
        )

    # --------------------------------------------------------
    # まずnavigation各ページから最低1chunk
    # --------------------------------------------------------

    for _candidate, docs in (
        navigation_doc_groups
    ):

        doc = docs[0]

        key = document_key(
            doc
        )

        if key in seen_document_keys:
            continue

        seen_document_keys.add(
            key
        )

        selected.append(
            doc
        )

        if len(
            selected
        ) >= CONTEXT_MAX_DOCS:
            return selected

    # --------------------------------------------------------
    # navigationページの残りchunk
    # --------------------------------------------------------

    for _candidate, docs in (
        navigation_doc_groups
    ):

        for doc in docs[1:]:

            key = document_key(
                doc
            )

            if key in seen_document_keys:
                continue

            seen_document_keys.add(
                key
            )

            selected.append(
                doc
            )

            if len(
                selected
            ) >= CONTEXT_MAX_DOCS:
                return selected

    # --------------------------------------------------------
    # 通常Hybrid Search
    # --------------------------------------------------------

    for doc in hybrid_documents:

        key = document_key(
            doc
        )

        if key in seen_document_keys:
            continue

        seen_document_keys.add(
            key
        )

        selected.append(
            doc
        )

        if len(
            selected
        ) >= CONTEXT_MAX_DOCS:
            break

    return selected


# ============================================================
# Navigation diagnostics
# ============================================================

def build_navigation_reason_map(
    navigation_pages: list,
):
    result = defaultdict(
        list
    )

    for candidate in navigation_pages:

        key = (
            candidate[
                "book"
            ],
            candidate[
                "pdf_page"
            ],
        )

        label = candidate.get(
            "label"
        )

        if (
            label
            and label
            not in result[
                key
            ]
        ):
            result[
                key
            ].append(
                label
            )

    return result


# ============================================================
# /ask
# ============================================================

@app.post(
    "/ask",
    response_model=QueryResponse,
)
def ask_question(
    request: QueryRequest,
):

    question = (
        request.question
    )

    books = (
        request.books
    )

    model_name = (
        request.model
        or "gpt-5.4-nano"
    )

    mode = (
        request.mode
        or "rules_strict"
    )

    initial_k = max(
        1,
        int(
            request.k
            or 10
        ),
    )

    max_k = (
        HYBRID_CANDIDATE_K
    )


    # ========================================================
    # free_chat
    # ========================================================

    if mode == "free_chat":

        system_prompt = (
            "あなたはソード・ワールド2.5の"
            "世界観とルールに精通したAIです。"
            "ユーザーの質問には必ずSW2.5の文脈で、"
            "具体的かつ専門的に回答してください。"
        )

        llm = ChatOpenAI(
            model_name=(
                model_name
            ),
            temperature=0.7,
        )

        messages = [
            SystemMessage(
                content=(
                    system_prompt
                )
            ),
            HumanMessage(
                content=(
                    question
                )
            ),
        ]

        response = llm.invoke(
            messages
        )

        return QueryResponse(
            answer=(
                response.content
            ),
            citations=[],
            sources=[],
            model_used=(
                model_name
            ),
            k_used=0,
            hybrid_k_used=0,
            navigation_pages_used=0,
            max_k=0,
            token_usage=(
                response
                .response_metadata
                .get(
                    "token_usage",
                    {},
                )
            ),
        )


    # ========================================================
    # exact_search
    # ========================================================

    elif mode == "exact_search":

        keywords = (
            question
            .strip()
            .split()
        )

        results = []

        for doc in search_documents:

            if (
                books
                and doc.metadata.get(
                    "book"
                )
                not in books
            ):
                continue

            if all(
                keyword
                in doc.page_content
                for keyword
                in keywords
            ):
                results.append(
                    doc
                )

        results = apply_category_weight(
            results
        )

        citations = build_citations(
            results
        )

        sources = (
            citations_to_legacy_sources(
                citations
            )
        )

        if sources:

            answer = (
                "全文検索を実施しました。"
                "結果は出典に記載されています。"
            )

        else:

            answer = (
                "該当はありませんでした。"
            )

        return QueryResponse(
            answer=answer,
            citations=citations,
            sources=sources,
            model_used=(
                "AIは使用していません"
            ),
            k_used=0,
            hybrid_k_used=0,
            navigation_pages_used=0,
            max_k=0,
            token_usage={},
        )


    # ========================================================
    # rules_strict
    # ========================================================

    else:

        # ----------------------------------------------------
        # 通常Hybrid Search
        #
        # kはこの枠だけに適用する。
        # ----------------------------------------------------

        hybrid_docs = (
            hybrid_search(
                query=question,
                top_k=(
                    initial_k
                ),
                candidate_k=(
                    max_k
                ),
                books=books,
            )
        )

        # ----------------------------------------------------
        # Navigation Search
        #
        # kとは独立した必須ページ枠。
        # ----------------------------------------------------

        (
            navigation_pages,
            _exact_index_match,
        ) = navigation_search(
            query=question,
            books=books,
        )

        # ----------------------------------------------------
        # Final context
        # ----------------------------------------------------

        selected_docs = (
            build_context_documents(
                question=(
                    question
                ),
                navigation_pages=(
                    navigation_pages
                ),
                hybrid_documents=(
                    hybrid_docs
                ),
            )
        )

        if not selected_docs:

            return QueryResponse(
                answer=(
                    "該当する情報が"
                    "見つかりませんでした。"
                ),
                citations=[],
                sources=[],
                model_used=(
                    model_name
                ),
                k_used=0,
                hybrid_k_used=0,
                navigation_pages_used=0,
                max_k=(
                    max_k
                ),
                token_usage={},
            )

        # ----------------------------------------------------
        # Citations
        # ----------------------------------------------------

        citations = (
            build_citations(
                selected_docs
            )
        )

        # Citationはbook + PDF pageで識別する。
        #
        # 表示ページはlogicalなので、
        # logicalだけをkeyにすると
        # front matter等で曖昧になり得る。
        citation_id_map = {
            (
                citation.book,
                citation.pdf_page,
            ): citation.id
            for citation
            in citations
        }

        navigation_reason_map = (
            build_navigation_reason_map(
                navigation_pages
            )
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context_parts = []

        for doc in selected_docs:

            book = doc.metadata.get(
                "book",
                "不明",
            )

            logical_page = (
                get_logical_page(
                    doc
                )
            )

            pdf_page = (
                get_pdf_page(
                    doc
                )
            )

            if (
                logical_page is None
                or pdf_page is None
            ):
                continue

            citation_id = (
                citation_id_map.get(
                    (
                        book,
                        pdf_page,
                    )
                )
            )

            if citation_id is None:
                continue

            nav_reasons = (
                navigation_reason_map.get(
                    (
                        book,
                        pdf_page,
                    ),
                    [],
                )
            )

            navigation_text = ""

            if nav_reasons:

                navigation_text = (
                    "\n検索補助情報: "
                    + " / ".join(
                        nav_reasons
                    )
                )

            context_parts.append(
                (
                    f"[CITATION:C{citation_id}]\n"
                    f"書籍: {book}\n"
                    f"書籍ページ: "
                    f"{logical_page}"
                    f"{navigation_text}\n"
                    f"本文:\n"
                    f"{doc.page_content}"
                )
            )

        context = "\n\n".join(
            context_parts
        )

        full_prompt = (
            prompt.format(
                context=context,
                question=question,
            )
        )

        llm = ChatOpenAI(
            model_name=(
                model_name
            ),
            temperature=0,
        )

        response = llm.invoke(
            full_prompt
        )

        sources = (
            citations_to_legacy_sources(
                citations
            )
        )

        return QueryResponse(
            answer=(
                response.content
            ),
            citations=(
                citations
            ),
            sources=(
                sources
            ),
            model_used=(
                model_name
            ),

            # contextに実際に投入されたchunk数
            k_used=(
                len(
                    selected_docs
                )
            ),

            # UI指定のHybrid枠
            hybrid_k_used=(
                len(
                    hybrid_docs
                )
            ),

            # kとは別枠
            navigation_pages_used=(
                len(
                    navigation_pages
                )
            ),

            max_k=(
                max_k
            ),

            token_usage=(
                response
                .response_metadata
                .get(
                    "token_usage",
                    {},
                )
            ),
        )