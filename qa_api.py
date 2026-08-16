from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

import json
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cloudflare_auth import (
    CloudflareAccessError,
    verify_cloudflare_access_token,
)
from user_store import (
    LastAdminError,
    UserAlreadyExistsError,
    UserNotFoundError,
    UserStoreError,
    create_user,
    ensure_user,
    get_user,
    init_user_store,
    list_users,
    touch_user_last_seen,
    update_user,
)

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage


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

# 検索再現率を維持しつつ、表・参照先・複数書籍を同時に入れられるよう少し余裕を持たせる。
CONTEXT_MAX_DOCS = int(os.getenv("CONTEXT_MAX_DOCS", "28"))

NAV_INDEX_FUZZY_THRESHOLD = float(
    os.getenv("NAV_INDEX_FUZZY_THRESHOLD", "0.72")
)
NAV_TOC_FUZZY_THRESHOLD = float(
    os.getenv("NAV_TOC_FUZZY_THRESHOLD", "0.60")
)
CITATION_EXCERPT_CHARS = int(os.getenv("CITATION_EXCERPT_CHARS", "360"))

# Query expansion
QUERY_VARIANT_MAX = int(os.getenv("QUERY_VARIANT_MAX", "6"))

# 表・一覧・チャート検索
STRUCTURED_MAX_PAGES = int(os.getenv("STRUCTURED_MAX_PAGES", "8"))
STRUCTURED_PAGES_PER_BOOK = int(os.getenv("STRUCTURED_PAGES_PER_BOOK", "2"))
STRUCTURED_CHUNKS_PER_PAGE = int(os.getenv("STRUCTURED_CHUNKS_PER_PAGE", "2"))
# 構造化データページは、表の途中がchunk境界で欠けないよう本文全体をcontextへ入れる。
STRUCTURED_CONTEXT_MAX_CHARS = int(os.getenv("STRUCTURED_CONTEXT_MAX_CHARS", "6000"))
STRUCTURED_FULL_PAGE_MAX = int(os.getenv("STRUCTURED_FULL_PAGE_MAX", "3"))

# 本文中の「⇒161頁」「次頁」等を追跡する。
REFERENCE_EXPAND_MAX_PAGES = int(os.getenv("REFERENCE_EXPAND_MAX_PAGES", "8"))
REFERENCE_PAGES_PER_SEED = int(os.getenv("REFERENCE_PAGES_PER_SEED", "2"))
REFERENCE_CHUNKS_PER_PAGE = int(os.getenv("REFERENCE_CHUNKS_PER_PAGE", "2"))

# 回答本文では未使用でも、調査価値が高い関連ページを出典欄へ残す。
SUPPLEMENTAL_CITATION_MAX = int(os.getenv("SUPPLEMENTAL_CITATION_MAX", "6"))
SUPPLEMENTAL_GENERAL_MAX = int(os.getenv("SUPPLEMENTAL_GENERAL_MAX", "3"))

NAVIGATION_REQUIRED = os.getenv("NAVIGATION_REQUIRED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


init_user_store()


# ============================================================
# Domain query expansion
# ============================================================

# ルール上の一般語とカテゴリ表記の橋渡し。
# ここは検索語の補助であり、回答内容そのものをハードコードするものではない。
DOMAIN_ALIASES = {
    "槍": ("スピア", "〈スピア〉"),
    "スピア": ("槍", "〈スピア〉"),
    "剣": ("ソード", "〈ソード〉"),
    "ソード": ("剣", "〈ソード〉"),
    "斧": ("アックス", "〈アックス〉"),
    "アックス": ("斧", "〈アックス〉"),
    "杖": ("スタッフ", "〈スタッフ〉"),
    "スタッフ": ("杖", "〈スタッフ〉"),
    "投擲": ("投擲武器", "〈投擲〉"),
    "流派": ("秘伝", "流派秘伝"),
    "秘伝": ("流派", "流派秘伝"),
    "両手持ち": ("2H", "1H両", "用法"),
    "両手": ("2H", "1H両"),
    "経験点テーブル": ("経験点表", "経験点", "テーブルA", "テーブルB"),
    "経験点表": ("経験点テーブル", "経験点"),
}

STRUCTURED_QUERY_TERMS = (
    "表",
    "一覧",
    "テーブル",
    "チャート",
    "早見表",
    "リスト",
)

QUERY_STOP_PHRASES = (
    "について詳しく教えてください",
    "について詳しく教えて",
    "について教えてください",
    "について教えて",
    "を詳しく教えてください",
    "を詳しく教えて",
    "を教えてください",
    "を教えて",
    "について説明してください",
    "について説明して",
    "を説明してください",
    "を説明して",
    "に関する情報を教えてください",
    "に関する情報を教えて",
    "に関する情報",
    "のルールについて",
    "のルール",
    "はありますでしょうか",
    "はありますか",
    "がありますでしょうか",
    "がありますか",
    "はあるでしょうか",
    "はあるか",
    "ありますでしょうか",
    "ありますか",
    "とは何ですか",
    "とは何",
    "って何ですか",
    "って何",
    "とは",
)


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
page_documents_by_logical = defaultdict(list)
logical_to_pdf = {}

for doc in search_documents:
    book = doc.metadata.get("book")
    pdf_page = doc.metadata.get("pdf_page", doc.metadata.get("page"))
    logical_page = doc.metadata.get("logical_page")
    if not book or pdf_page is None or logical_page is None:
        continue
    try:
        pdf_page = int(pdf_page)
        logical_page = int(logical_page)
    except (TypeError, ValueError):
        continue

    page_documents_by_pdf[(book, pdf_page)].append(doc)
    page_documents_by_logical[(book, logical_page)].append(doc)
    logical_to_pdf[(book, logical_page)] = pdf_page

for docs in page_documents_by_pdf.values():
    docs.sort(key=lambda doc: int(doc.metadata.get("chunk", 0)))
for docs in page_documents_by_logical.values():
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


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    books: Optional[List[str]] = None
    model: Optional[str] = "gpt-5.4-nano"
    k: Optional[int] = 20
    mode: Optional[str] = "rules_strict"
    history: Optional[List[ChatMessage]] = None


class Citation(BaseModel):
    id: int
    book: str
    page: int
    pdf_page: int
    category: Optional[str] = None
    excerpt: Optional[str] = None
    reason: Optional[str] = None
    # True: 回答本文で[Cx]として引用、False: 調査価値の高い関連資料
    used_in_answer: bool = True


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    model_used: Optional[str] = None
    k_used: Optional[int] = None
    hybrid_k_used: Optional[int] = None
    navigation_pages_used: Optional[int] = None
    structured_pages_used: Optional[int] = None
    reference_pages_used: Optional[int] = None
    max_k: Optional[int] = None
    token_usage: Optional[dict] = None


# ============================================================
# Authentication / authorization
# ============================================================


class AuthMeResponse(BaseModel):
    email: str
    display_name: str
    is_admin: bool


class AdminUserResponse(BaseModel):
    email: str
    display_name: str
    is_admin: bool
    created_at: str
    updated_at: str
    last_seen_at: Optional[str] = None


class AdminUserCreateRequest(BaseModel):
    email: str
    display_name: str = ""
    is_admin: bool = False


class AdminUserUpdateRequest(BaseModel):
    current_email: str
    email: str
    display_name: str = ""
    is_admin: bool = False


def get_verified_cloudflare_email(
    cf_access_jwt_assertion: str | None,
) -> str:
    if not cf_access_jwt_assertion:
        raise HTTPException(
            status_code=403,
            detail="Cf-Access-Jwt-Assertion header is missing",
        )

    try:
        payload = verify_cloudflare_access_token(cf_access_jwt_assertion)
    except CloudflareAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    return payload["email"]


def get_authorized_user(
    cf_access_jwt_assertion: str | None,
    *,
    require_admin: bool = False,
) -> dict:
    email = get_verified_cloudflare_email(cf_access_jwt_assertion)
    try:
        user = ensure_user(email)
    except UserStoreError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ユーザー情報の初期化に失敗しました: {exc}",
        ) from exc
    if require_admin and not user["is_admin"]:
        raise HTTPException(
            status_code=403,
            detail="この操作には管理者権限が必要です。",
        )

    return user


def _admin_user_response(user: dict) -> AdminUserResponse:
    return AdminUserResponse(
        email=user["email"],
        display_name=user["display_name"],
        is_admin=bool(user["is_admin"]),
        created_at=user["created_at"],
        updated_at=user["updated_at"],
        last_seen_at=user.get("last_seen_at"),
    )


@app.get("/auth/me", response_model=AuthMeResponse)
def auth_me(
    cf_access_jwt_assertion: str | None = Header(
        default=None,
        alias="Cf-Access-Jwt-Assertion",
    ),
):
    user = get_authorized_user(cf_access_jwt_assertion)
    touch_user_last_seen(user["email"])
    return AuthMeResponse(
        email=user["email"],
        display_name=user["display_name"],
        is_admin=bool(user["is_admin"]),
    )


@app.get("/admin/users", response_model=List[AdminUserResponse])
def admin_list_users(
    cf_access_jwt_assertion: str | None = Header(
        default=None,
        alias="Cf-Access-Jwt-Assertion",
    ),
):
    get_authorized_user(
        cf_access_jwt_assertion,
        require_admin=True,
    )
    return [_admin_user_response(user) for user in list_users()]


@app.post("/admin/users", response_model=AdminUserResponse)
def admin_create_user(
    request: AdminUserCreateRequest,
    cf_access_jwt_assertion: str | None = Header(
        default=None,
        alias="Cf-Access-Jwt-Assertion",
    ),
):
    get_authorized_user(
        cf_access_jwt_assertion,
        require_admin=True,
    )
    try:
        user = create_user(
            request.email,
            request.display_name,
            is_admin=request.is_admin,
            is_active=True,
        )
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except UserStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _admin_user_response(user)


@app.put("/admin/users", response_model=AdminUserResponse)
def admin_update_user(
    request: AdminUserUpdateRequest,
    cf_access_jwt_assertion: str | None = Header(
        default=None,
        alias="Cf-Access-Jwt-Assertion",
    ),
):
    actor = get_authorized_user(
        cf_access_jwt_assertion,
        require_admin=True,
    )

    actor_email = actor["email"].strip().lower()
    target_email = request.current_email.strip().lower()
    new_email = request.email.strip().lower()

    if target_email == actor_email:
        if new_email != actor_email:
            raise HTTPException(
                status_code=400,
                detail="現在ログイン中の管理者自身のメールアドレスは変更できません。",
            )
        if not request.is_admin:
            raise HTTPException(
                status_code=400,
                detail="現在ログイン中の管理者自身の管理者権限は解除できません。",
            )

    try:
        user = update_user(
            request.current_email,
            email=request.email,
            display_name=request.display_name,
            is_admin=request.is_admin,
        )
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LastAdminError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UserStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _admin_user_response(user)


# ============================================================
# Prompt
# ============================================================

template = """
以下のコンテキストだけを根拠として、質問に正確に答えてください。

回答ルール:
- 「これまでの会話」は、現在の質問の文脈や指示語を理解するためだけに使用してください。
- 過去のAI回答を、ルール・数値・条件・例外などの根拠として扱ってはいけません。
- 事実関係は、今回取得したコンテキストで確認できた情報だけを根拠にしてください。
- コンテキストに存在しない情報を推測して補ってはいけません。
- 根拠を示す場合は、対応するコンテキストの引用IDを `[C1]` の形式で記載してください。
- 引用IDは必ずコンテキスト中に存在するものだけを使用してください。
- `[C1]`、`[C2]` のような形式以外で出典を書いてはいけません。
- 書籍名やページ番号を回答文中へ直接書く必要はありません。
- 出典は、その出典によって裏付けられる文章または段落の末尾に記載してください。
- 同一段落内では、同じ引用IDを繰り返してはいけません。
- 一つの段落が複数の引用元に基づく場合は、段落末尾に `[C1][C2]` のようにまとめて記載してください。
- 長い回答では、どの記述がどの根拠に基づくか判別できるよう、必要な段落ごとに引用してください。
- 「索引」や「目次」は検索の手掛かりであり、それ自体をルール本文として扱わないでください。
- 本文に「○頁参照」「次頁」などがあり、その参照先がコンテキストに含まれる場合は、参照先の内容も確認して回答してください。
- 同じ事項について複数の書籍に記載がある場合、基本ルールブックに基本的な定義・数値・種族特徴・ルール本文が存在するなら、原則としてそれを回答の基礎にしてください。
- サプリメントや追加書籍の記述は、基本ルールを置き換えるものと明記されていない限り、追加情報・補足情報として扱ってください。
- 質問が希少種、追加種族、追加技能、追加魔法、追加アイテム、追加戦闘特技など、特定の追加要素を明示している場合は、その要素を収録したサプリメント側の記述を優先してください。
- 基本種と希少種、基本ルールと追加ルールなど、異なる対象を混同しないでください。
- 表・テーブル・一覧・チャートを求める質問では、原則として表そのものを回答本文へ転記する必要はありません。何の表か、どの範囲を収録しているか、どの出典を参照すべきかを簡潔に説明してください。正確な数値・項目は出典ページを参照できるようにしてください。
- 表・テーブル・一覧・チャートが複数書籍にある場合は、最初に見つかった1冊だけで回答を終えず、各表の収録範囲を確認してください。より完全・拡張された表がある場合は、それが存在することと収録範囲を回答で明示してください。簡略版・初心者向け・追加範囲の表も、参照価値があれば区別して示してください。
- 同一テーマについて複数書籍に直接関係する規定・例外・追加ルールがコンテキストにある場合、質問が横断的な情報収集を求めているなら、基本ルールだけで回答を打ち切らず、重複を避けつつ異なる内容を拾ってください。
- 「○○を使った流派はあるか」のような存在確認では、直接一致だけでなく、同義のカテゴリ表記や「流派」「秘伝」など関連する本文から具体例を探してください。
- 十分な根拠がコンテキストにない場合は、その旨を明確に回答してください。

今回の回答方針:
{answer_guidance}

これまでの会話:
{conversation_history}

コンテキスト:
{context}

質問:
{question}
"""

prompt = PromptTemplate(
    input_variables=[
        "context",
        "question",
        "answer_guidance",
        "conversation_history",
    ],
    template=template,
)


def normalize_chat_history(
    history: Optional[List[ChatMessage]],
    max_messages: int = 12,
) -> list[dict]:
    result = []
    for message in history or []:
        role = str(message.role or "").strip().lower()
        content = str(message.content or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        result.append({"role": role, "content": content})
    return result[-max_messages:]


def format_conversation_history(history: list[dict]) -> str:
    if not history:
        return "（なし）"

    lines = []
    for message in history:
        label = "ユーザー" if message["role"] == "user" else "アシスタント"
        lines.append(f"{label}: {message['content']}")
    return "\n\n".join(lines)


def build_contextual_search_question(
    question: str,
    history: list[dict],
) -> str:
    """
    掘り下げ質問の「それ」「その場合」などを検索しやすくする。

    AI回答本文は検索語へ混ぜず、直近のユーザー質問だけを補助文脈として使う。
    """
    previous_questions = [
        message["content"]
        for message in history
        if message["role"] == "user"
    ][-3:]

    if not previous_questions:
        return question

    parts = previous_questions + [question]
    search_question = "\n".join(
        part.strip() for part in parts if part and part.strip()
    )
    return search_question[-1600:]


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


def page_pdf_from_logical(book: str, logical_page: int) -> int | None:
    return logical_to_pdf.get((book, logical_page))


def full_page_text(book: str, pdf_page: int) -> str:
    return "\n".join(
        doc.page_content or ""
        for doc in page_documents_by_pdf.get((book, pdf_page), [])
    )


def structured_context_text(book: str, pdf_page: int, question: str) -> str:
    """表・一覧ページはchunk境界で欠けないよう、ページ本文をまとめて返す。"""
    text = full_page_text(book, pdf_page).strip()
    if len(text) <= STRUCTURED_CONTEXT_MAX_CHARS:
        return text

    terms = [normalize_search_query(question)] + extract_query_terms(question)
    positions = [text.find(term) for term in terms if term and text.find(term) >= 0]
    if positions:
        center = min(positions)
        start = max(0, center - STRUCTURED_CONTEXT_MAX_CHARS // 5)
    else:
        start = 0
    end = min(len(text), start + STRUCTURED_CONTEXT_MAX_CHARS)
    if end - start < STRUCTURED_CONTEXT_MAX_CHARS:
        start = max(0, end - STRUCTURED_CONTEXT_MAX_CHARS)
    excerpt = text[start:end]
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(text):
        excerpt += "…"
    return excerpt


# ============================================================
# Query normalization / expansion / text scoring
# ============================================================


def normalize_search_query(query: str) -> str:
    normalized = query.strip()
    normalized = re.sub(r"[。．.!！?？]+$", "", normalized).strip()

    changed = True
    while changed:
        changed = False
        for phrase in QUERY_STOP_PHRASES:
            if normalized.endswith(phrase):
                normalized = normalized[: -len(phrase)].strip()
                changed = True
                break

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


def query_is_structured(question: str) -> bool:
    normalized = normalize_search_query(question)
    return any(term in normalized for term in STRUCTURED_QUERY_TERMS)


def query_requests_broad_coverage(question: str) -> bool:
    """複数資料を横断して規定・補足を集めるべき質問かを判定する。"""
    text = question.strip()
    broad_markers = (
        "に関する情報",
        "関連する情報",
        "関連情報",
        "ルールを教えて",
        "ルールについて",
        "規定を教えて",
        "扱いを教えて",
    )
    return any(marker in text for marker in broad_markers)


def build_answer_guidance(question: str) -> str:
    if query_is_structured(question):
        return (
            "この質問は表・テーブル・一覧・チャートを探しています。"
            "コンテキスト内の構造化データページを複数書籍ぶん確認してください。"
            "回答本文へ表の数値を転記することより、どの出典にどの範囲の表があるかを"
            "簡潔に案内することを優先してください。"
            "簡略版と完全版がある場合は、完全版・収録範囲の広い版を明示し、"
            "他書籍の表に追加範囲や差異があれば区別して案内してください。"
        )
    if query_requests_broad_coverage(question):
        return (
            "この質問は同一テーマに関する情報を横断的に求めています。"
            "基本ルールを最初に示したうえで、コンテキスト中の別書籍に"
            "直接関係する追加規定・例外・補足があるか確認し、"
            "単なる重複は省きつつ、異なる内容は回答へ含めてください。"
        )
    return (
        "質問対象に直接対応する規定を優先し、基本資料と追加資料の役割を区別して"
        "必要十分な範囲で回答してください。"
    )


def extract_query_terms(query: str) -> list[str]:
    text = normalize_search_query(query)
    for phrase in (
        "を使った",
        "を使う",
        "を使用した",
        "を使用する",
        "に関する",
        "について",
        "という",
        "として",
    ):
        text = text.replace(phrase, " ")

    # 助詞は単独スペース化できる範囲だけ除く。
    text = re.sub(r"[、，,。]+", " ", text)
    terms = [value.strip() for value in re.split(r"\s+", text) if value.strip()]

    expanded = []
    for term in terms:
        # 「槍の流派」等の短い連結を補助的に分解する。
        parts = [p for p in re.split(r"の", term) if p]
        expanded.extend(parts or [term])

    result = []
    for term in expanded:
        if len(term) == 1 and term in {"は", "が", "を", "に", "で", "と"}:
            continue
        if term not in result:
            result.append(term)
    return result


def build_query_variants(question: str) -> list[str]:
    base = normalize_search_query(question)
    variants = []

    def add(value: str):
        value = re.sub(r"\s+", " ", value or "").strip()
        if value and value not in variants and len(variants) < QUERY_VARIANT_MAX:
            variants.append(value)

    add(base)

    terms = extract_query_terms(question)
    if len(terms) >= 2:
        add(" ".join(terms))

    # 個々の語にルール用語の別名を展開する。
    for index, term in enumerate(terms):
        aliases = DOMAIN_ALIASES.get(term, ())
        for alias in aliases:
            replaced = list(terms)
            replaced[index] = alias
            add(" ".join(replaced))

    # 文章中に直接含まれる別名辞書キーも置換する。
    for key, aliases in DOMAIN_ALIASES.items():
        if key not in base:
            continue
        for alias in aliases:
            add(base.replace(key, alias))

    return variants or [question.strip()]


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
    prefixes = (
        "索引: ",
        "目次: ",
        "目次セクション続き: ",
        "表・一覧検索: ",
    )
    for prefix in prefixes:
        if reason.startswith(prefix):
            value = reason[len(prefix) :].split(" /")[0].strip()
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


def build_excerpt_from_text(
    raw_text: str,
    query: str,
    reason: str = "",
    max_chars: int = CITATION_EXCERPT_CHARS,
) -> str:
    text = re.sub(r"\s+", " ", raw_text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    search_terms = [normalize_search_query(query)]
    search_terms.extend(extract_reason_terms(reason))
    search_terms.extend(extract_query_terms(query))
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


def build_excerpt(
    doc,
    query: str,
    reason: str = "",
    max_chars: int = CITATION_EXCERPT_CHARS,
) -> str:
    return build_excerpt_from_text(
        doc.page_content or "",
        query,
        reason,
        max_chars,
    )


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
                excerpt=build_excerpt_from_text(
                    best_item.get("context_text") or best_doc.page_content or "",
                    question,
                    reason,
                ),
                reason=reason or "検索結果から選定",
                used_in_answer=True,
            )
        )

    return citations


def citations_to_legacy_sources(citations: List[Citation]) -> List[str]:
    result = []
    for citation in citations:
        if citation.category:
            result.append(f"{citation.category} / {citation.book} - p.{citation.page}")
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


def collapse_repeated_citations(answer: str) -> str:
    """同一引用が連続する段落・箇条書きでは、最後の出現だけを残す。"""
    if not answer:
        return answer

    lines = answer.splitlines()
    citation_pattern = re.compile(r"(?:\s*\[C\d+\])+\s*$")
    citation_id_pattern = re.compile(r"\[C(\d+)\]")

    def line_citations(line: str) -> list[int]:
        match = citation_pattern.search(line)
        if not match:
            return []
        return [int(value) for value in citation_id_pattern.findall(match.group(0))]

    def strip_ids(line: str, ids: set[int]) -> str:
        if not ids:
            return line
        match = citation_pattern.search(line)
        if not match:
            return line
        suffix = match.group(0)
        kept = [
            int(value)
            for value in citation_id_pattern.findall(suffix)
            if int(value) not in ids
        ]
        base = line[: match.start()].rstrip()
        if kept:
            return base + " " + "".join(f"[C{value}]" for value in kept)
        return base

    # 空行で区切られない連続行を1ブロックとして扱う。
    block_start = 0
    while block_start < len(lines):
        while block_start < len(lines) and not lines[block_start].strip():
            block_start += 1
        if block_start >= len(lines):
            break

        block_end = block_start
        while block_end + 1 < len(lines) and lines[block_end + 1].strip():
            block_end += 1

        last_occurrence = {}
        for index in range(block_start, block_end + 1):
            for citation_id in line_citations(lines[index]):
                last_occurrence[citation_id] = index

        for index in range(block_start, block_end + 1):
            remove_ids = {
                citation_id
                for citation_id in line_citations(lines[index])
                if last_occurrence.get(citation_id) != index
            }
            lines[index] = strip_ids(lines[index], remove_ids)

        block_start = block_end + 1

    return "\n".join(lines)


def citation_query_relevance(citation: Citation, question: str) -> float:
    """未使用出典を返す際の最低限の質問関連度を評価する。"""
    query_terms = extract_query_terms(question)
    haystack = " ".join(
        value
        for value in (
            citation.excerpt or "",
            citation.reason or "",
        )
        if value
    )
    normalized_haystack = normalize_navigation_text(haystack)
    score = 0.0

    normalized_query = normalize_navigation_text(normalize_search_query(question))
    if normalized_query and normalized_query in normalized_haystack:
        score += 30.0

    for term in query_terms:
        normalized_term = normalize_navigation_text(term)
        if not normalized_term:
            continue
        if normalized_term in normalized_haystack:
            score += 12.0
        else:
            score += char_ngram_coverage(
                normalized_term,
                normalized_haystack,
                2,
            ) * 3.0
    return score


def select_return_citations(
    answer: str,
    citations: List[Citation],
    context_items: list[dict],
    structured_query: bool,
    question: str,
) -> List[Citation]:
    used_ids = extract_used_citation_ids(answer)
    used_set = set(used_ids)

    citation_by_id = {citation.id: citation for citation in citations}
    result = []

    # 回答で実際に使われた出典は回答中の出現順に返す。
    for citation_id in used_ids:
        citation = citation_by_id.get(citation_id)
        if citation is None:
            continue
        citation.used_in_answer = True
        result.append(citation)

    # ページごとのcontext score/sourceを取得する。
    page_meta = {}
    for item in context_items:
        doc = item["doc"]
        key = page_key_from_doc(doc)
        if key is None:
            continue
        current = page_meta.get(key)
        candidate = {
            "score": item.get("context_score", 0.0),
            "source": item.get("source", ""),
        }
        if current is None or candidate["score"] > current["score"]:
            page_meta[key] = candidate

    # 表・一覧質問では、回答本文で未引用でも複数書籍の表本体を残す。
    # 通常質問でも、参照先やnavigation由来の高価値ページを少数残す。
    supplemental_limit = (
        SUPPLEMENTAL_CITATION_MAX if structured_query else SUPPLEMENTAL_GENERAL_MAX
    )

    source_bonus = {
        # 表・一覧質問では、未引用でも実際の表本体を最優先で残す。
        "structured": 220.0 if structured_query else 70.0,
        "reference": 30.0 if structured_query else 80.0,
        "index": 50.0,
        "toc": 35.0,
        "section": 25.0,
        "hybrid": 0.0,
    }

    candidates = []
    for citation in citations:
        if citation.id in used_set:
            continue
        meta = page_meta.get((citation.book, citation.pdf_page), {})
        source = meta.get("source", "")

        # 通常質問では純粋なhybrid未引用ページは出典欄に残さない。
        if not structured_query and source == "hybrid":
            continue

        relevance = citation_query_relevance(citation, question)
        # navigation/reference由来でも、質問語との関連がほぼ確認できないページは
        # supplemental citationとして返さない。
        if not structured_query and relevance < 4.0:
            continue

        score = (
            float(meta.get("score", 0.0))
            + source_bonus.get(source, 0.0)
            + relevance
        )
        candidates.append((citation, score, source))

    candidates.sort(key=lambda item: item[1], reverse=True)

    # structured queryでは、まず表本体を別書籍から優先的に確保する。
    if structured_query:
        structured_candidates = [
            item for item in candidates if item[2] == "structured"
        ]
        other_candidates = [
            item for item in candidates if item[2] != "structured"
        ]
        candidates = structured_candidates + other_candidates

    # structured queryでは書籍の多様性を優先する。
    selected_supplemental = []
    seen_books = set(citation.book for citation in result)
    if structured_query:
        for citation, score, source in candidates:
            if len(selected_supplemental) >= supplemental_limit:
                break
            if citation.book in seen_books:
                continue
            selected_supplemental.append((citation, score, source))
            seen_books.add(citation.book)

    for citation, score, source in candidates:
        if len(selected_supplemental) >= supplemental_limit:
            break
        if any(existing.id == citation.id for existing, _, _ in selected_supplemental):
            continue
        selected_supplemental.append((citation, score, source))

    for citation, _score, _source in selected_supplemental:
        citation.used_in_answer = False
        result.append(citation)

    return result


# ============================================================
# Search helpers
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


def hybrid_search(
    query: str,
    top_k: int,
    candidate_k: int,
    books=None,
    variants: Optional[list[str]] = None,
):
    query_variants = variants or build_query_variants(query)

    # 複数query variantのRRFを一つのdoc scoreへ統合する。
    scores_by_key = defaultdict(float)
    docs_by_key = {}
    reasons_by_key = defaultdict(list)
    rrf_k = 60.0
    vector_weight = 0.40
    lexical_weight = 0.60

    for variant_no, search_query in enumerate(query_variants, start=1):
        vector_results = db.similarity_search_with_score(search_query, k=candidate_k)
        vector_position = 0
        for doc, _distance in vector_results:
            if get_logical_page(doc) is None:
                continue
            if books and doc.metadata.get("book") not in books:
                continue
            vector_position += 1
            key = document_key(doc)
            docs_by_key[key] = doc
            scores_by_key[key] += vector_weight / (rrf_k + vector_position)
            if vector_position <= 15:
                reasons_by_key[key].append(
                    f"ベクトル検索 #{vector_position}（{search_query}）"
                )

        lexical_results = lexical_search(search_query, candidate_k, books=books)
        for lexical_position, (doc, _score) in enumerate(lexical_results, start=1):
            key = document_key(doc)
            docs_by_key[key] = doc
            scores_by_key[key] += lexical_weight / (rrf_k + lexical_position)
            if lexical_position <= 15:
                reasons_by_key[key].append(
                    f"文字列検索 #{lexical_position}（{search_query}）"
                )

    scored_candidates = []
    for key, score in scores_by_key.items():
        doc = docs_by_key[key]
        # variantを複数拾う文書は自然にRRF加点される。
        score *= get_document_authority(doc)
        scored_candidates.append((doc, score, reasons_by_key[key]))

    scored_candidates.sort(key=lambda item: item[1], reverse=True)

    # 定義ページはprimary + alias variantsから少量だけ先行採用する。
    definition_candidates = []
    definition_seen = set()
    for variant in query_variants[:3]:
        for doc, score in definition_search(variant, top_k=2, books=books):
            key = document_key(doc)
            if key in definition_seen:
                continue
            definition_seen.add(key)
            definition_candidates.append((doc, score, variant))

    definition_candidates.sort(key=lambda item: item[1], reverse=True)

    selected = []
    seen = set()

    for doc, score, variant in definition_candidates:
        key = document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "doc": doc,
                "retrieval_score": score,
                "reason": f"定義候補として直接一致（{variant}）",
            }
        )
        if len(selected) >= top_k:
            return selected

    for doc, score, reasons in scored_candidates:
        key = document_key(doc)
        if key in seen:
            continue
        seen.add(key)
        short_reasons = reasons[:3]
        reason = "通常検索で高関連"
        if short_reasons:
            reason += "（" + " / ".join(short_reasons) + "）"
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


def build_section_expansion(toc_candidates: list):
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


def navigation_search(query: str, books=None, variants: Optional[list[str]] = None):
    query_variants = variants or build_query_variants(query)
    merged = {}
    any_exact = False

    source_priority = {"index": 3, "toc": 2, "section": 1}

    for variant in query_variants:
        index_candidates, exact_index_match = navigation_index_search(variant, books=books)
        toc_candidates = navigation_toc_search(variant, books=books)
        any_exact = any_exact or exact_index_match

        section_candidates = []
        # そのvariantでexact indexがある場合は固有項目検索とみなし、章全体の展開を抑える。
        if not exact_index_match:
            section_candidates = build_section_expansion(toc_candidates)

        for candidate in index_candidates + toc_candidates + section_candidates:
            candidate = dict(candidate)
            if variant != normalize_search_query(query):
                candidate["label"] = f"{candidate['label']}（検索展開: {variant}）"
                candidate["score"] = candidate.get("score", 0.0) - 1.5

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
    return candidates[:NAV_MANDATORY_MAX_PAGES], any_exact


# ============================================================
# Structured-data page search
# ============================================================


def structured_page_search(
    question: str,
    books=None,
    variants: Optional[list[str]] = None,
) -> list[dict]:
    if not query_is_structured(question):
        return []

    query_variants = variants or build_query_variants(question)
    normalized_variants = [
        normalize_navigation_text(value) for value in query_variants if value
    ]
    normalized_primary = normalize_navigation_text(normalize_search_query(question))

    grouped = {}
    for doc in search_documents:
        book = doc.metadata.get("book")
        if books and book not in books:
            continue
        pdf_page = get_pdf_page(doc)
        logical_page = get_logical_page(doc)
        if not book or pdf_page is None or logical_page is None:
            continue

        text = doc.page_content or ""
        normalized_text = normalize_navigation_text(text)
        if not normalized_text:
            continue

        score = 0.0
        matched_terms = []

        if normalized_primary and normalized_primary in normalized_text:
            score += 80.0
            matched_terms.append(normalize_search_query(question))

        for variant, normalized_variant in zip(query_variants, normalized_variants):
            if not normalized_variant:
                continue
            if normalized_variant in normalized_text:
                score += 40.0
                matched_terms.append(variant)
            else:
                coverage = char_ngram_coverage(normalized_variant, normalized_text, 2)
                if coverage >= 0.65:
                    score += coverage * 18.0

        structured_hits = sum(
            1 for term in STRUCTURED_QUERY_TERMS if term in text
        )
        score += structured_hits * 8.0

        # 表本体は数字・レベル・A/B等の密度が高い傾向を利用する。
        digit_count = sum(ch.isdigit() for ch in text)
        if len(text) > 0:
            digit_ratio = digit_count / len(text)
            score += min(18.0, digit_ratio * 180.0)

        for marker in ("レベル", "必要経験点", "テーブルA", "テーブルB", "合計", "経験点"):
            if marker in text:
                score += 4.0

        # 表の導入文だけでなく表本体を優先するため、十分な構造信号がないページは弱くする。
        if score < 25.0:
            continue

        score *= get_document_authority(doc)
        key = (book, pdf_page)
        existing = grouped.get(key)
        item = {
            "book": book,
            "pdf_page": pdf_page,
            "logical_page": logical_page,
            "source": "structured",
            "score": score,
            "authority": get_book_authority(book),
            "label": "表・一覧検索: " + (
                matched_terms[0] if matched_terms else normalize_search_query(question)
            ),
        }
        if existing is None or item["score"] > existing["score"]:
            grouped[key] = item

    candidates = sorted(
        grouped.values(),
        key=lambda item: (item["score"], item["authority"]),
        reverse=True,
    )

    # 同じ書籍の似た表だけで埋まらないよう、まず1冊1ページずつ採る。
    selected = []
    selected_keys = set()
    per_book = defaultdict(int)

    for candidate in candidates:
        if len(selected) >= STRUCTURED_MAX_PAGES:
            break
        book = candidate["book"]
        if per_book[book] > 0:
            continue
        selected.append(candidate)
        selected_keys.add((book, candidate["pdf_page"]))
        per_book[book] += 1

    for candidate in candidates:
        if len(selected) >= STRUCTURED_MAX_PAGES:
            break
        key = (candidate["book"], candidate["pdf_page"])
        if key in selected_keys:
            continue
        if per_book[candidate["book"]] >= STRUCTURED_PAGES_PER_BOOK:
            continue
        selected.append(candidate)
        selected_keys.add(key)
        per_book[candidate["book"]] += 1

    return selected


# ============================================================
# Reference expansion
# ============================================================


def extract_page_references(text: str, current_logical_page: int) -> list[tuple[int, str]]:
    refs = []
    if not text:
        return refs

    patterns = [
        (r"[⇒→]\s*(\d{1,4})\s*頁", "矢印参照"),
        (r"(\d{1,4})\s*頁\s*(?:も|を)?\s*参照", "本文参照"),
        (r"(?:詳しくは|詳細は|については)[^。\n]{0,30}?(\d{1,4})\s*頁", "詳細参照"),
    ]

    for pattern, label in patterns:
        for match in re.finditer(pattern, text):
            try:
                page = int(match.group(1))
            except (TypeError, ValueError):
                continue
            snippet = match.group(0).strip()
            refs.append((page, snippet or label))

    if re.search(r"次頁|次ページ|次のページ", text):
        refs.append((current_logical_page + 1, "次頁"))
    if current_logical_page > 1 and re.search(r"前頁|前ページ|前のページ", text):
        refs.append((current_logical_page - 1, "前頁"))

    deduped = []
    seen = set()
    for page, label in refs:
        if page <= 0 or page in seen:
            continue
        seen.add(page)
        deduped.append((page, label))
    return deduped


def reference_expand_pages(seed_items: list[dict], books=None) -> list[dict]:
    candidates = {}

    for seed in seed_items:
        doc = seed["doc"]
        book = doc.metadata.get("book")
        pdf_page = get_pdf_page(doc)
        logical_page = get_logical_page(doc)
        if not book or pdf_page is None or logical_page is None:
            continue
        if books and book not in books:
            continue

        page_text = full_page_text(book, pdf_page)
        references = extract_page_references(page_text, logical_page)

        for ref_logical, ref_text in references[:REFERENCE_PAGES_PER_SEED]:
            ref_pdf = page_pdf_from_logical(book, ref_logical)
            if ref_pdf is None or ref_pdf == pdf_page:
                continue
            key = (book, ref_pdf)
            score = float(seed.get("context_score", 0.0)) + 150.0
            reason = (
                f"参照先補完: {book} p.{logical_page} の「{ref_text}」から "
                f"p.{ref_logical}"
            )
            item = {
                "book": book,
                "pdf_page": ref_pdf,
                "logical_page": ref_logical,
                "source": "reference",
                "score": score,
                "authority": get_book_authority(book),
                "label": reason,
            }
            existing = candidates.get(key)
            if existing is None or item["score"] > existing["score"]:
                candidates[key] = item

    result = sorted(
        candidates.values(),
        key=lambda item: (item["score"], item["authority"]),
        reverse=True,
    )
    return result[:REFERENCE_EXPAND_MAX_PAGES]


# ============================================================
# Context reranking / pruning
# ============================================================


def select_candidate_page_documents(
    candidate: dict,
    question: str,
    max_chunks: int,
    mandatory_first: bool = True,
) -> list[dict]:
    docs = list(
        page_documents_by_pdf.get((candidate["book"], candidate["pdf_page"]), [])
    )
    if not docs:
        return []

    prefer_page_start = candidate["source"] in {"toc", "section", "reference"}
    terms = extract_reason_terms(candidate.get("label", ""))
    terms.extend(extract_query_terms(question))
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
        key=lambda item: (item[1], -int(item[0].metadata.get("chunk", 0))),
        reverse=True,
    )

    source_base = {
        "reference": 1500.0,
        "structured": 1350.0,
        "index": 1200.0,
        "toc": 1100.0,
        "section": 1050.0,
    }.get(candidate["source"], 1000.0)

    selected = []
    for index, (doc, chunk_score) in enumerate(scored[:max_chunks]):
        authority_note = ""
        if get_document_authority(doc) > 1.0:
            authority_note = " / 資料優先度を加味"
        selected.append(
            {
                "doc": doc,
                "mandatory": mandatory_first and index == 0,
                "context_score": source_base + candidate.get("score", 0.0) + chunk_score,
                "reason": candidate.get("label", "検索補完") + authority_note,
                "source": candidate["source"],
            }
        )
    return selected


def build_context_items(
    *,
    question: str,
    navigation_pages: list,
    structured_pages: list,
    hybrid_items: list,
) -> tuple[list[dict], list[dict]]:
    navigation_items = []
    for candidate in navigation_pages:
        navigation_items.extend(
            select_candidate_page_documents(
                candidate,
                question,
                max_chunks=NAV_CHUNKS_PER_PAGE,
                mandatory_first=True,
            )
        )

    structured_items = []
    for structured_rank, candidate in enumerate(structured_pages, start=1):
        page_items = select_candidate_page_documents(
            candidate,
            question,
            max_chunks=max(1, STRUCTURED_CHUNKS_PER_PAGE),
            mandatory_first=True,
        )
        if not page_items:
            continue

        # 表はchunk境界で行や列が分断されやすい。
        # 構造化検索で選ばれたページは、代表chunk 1件にページ本文全体を
        # context_textとして持たせ、1ページ=1context itemとして扱う。
        best_item = page_items[0]
        if structured_rank <= STRUCTURED_FULL_PAGE_MAX:
            best_item["context_text"] = structured_context_text(
                candidate["book"],
                candidate["pdf_page"],
                question,
            )
            best_item["reason"] = (
                best_item.get("reason", "表・一覧検索")
                + " / 表本体をページ単位で採用"
            )
        structured_items.append(best_item)

    hybrid_context_items = []
    for rank, item in enumerate(hybrid_items, start=1):
        doc = item["doc"]
        relevance = chunk_relevance_score(
            doc,
            question,
            extra_terms=extract_query_terms(question),
        )
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

    # 参照先追跡のseedには、navigation/structured上位とhybrid上位を使う。
    reference_seed_items = (
        navigation_items
        + structured_items
        + hybrid_context_items[: min(12, len(hybrid_context_items))]
    )
    reference_pages = reference_expand_pages(reference_seed_items)

    reference_items = []
    for candidate in reference_pages:
        reference_items.extend(
            select_candidate_page_documents(
                candidate,
                question,
                max_chunks=REFERENCE_CHUNKS_PER_PAGE,
                mandatory_first=True,
            )
        )

    all_special_items = reference_items + structured_items + navigation_items
    mandatory_items = [item for item in all_special_items if item["mandatory"]]
    optional_special = [item for item in all_special_items if not item["mandatory"]]

    # reference > structured > navigationの順に最低1chunkを保護する。
    mandatory_items.sort(key=lambda item: item["context_score"], reverse=True)
    optional_pool = optional_special + hybrid_context_items
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

    for item in mandatory_items:
        if len(selected) >= CONTEXT_MAX_DOCS:
            break
        add_item(item)

    for item in optional_pool:
        if len(selected) >= CONTEXT_MAX_DOCS:
            break
        add_item(item)

    return selected, reference_pages


# ============================================================
# Exact-search helpers
# ============================================================


def apply_category_weight(results):
    return sorted(results, key=get_document_authority, reverse=True)


def build_exact_search_items(results, question: str) -> list[dict]:
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
    initial_k = max(1, int(request.k or 20))
    max_k = HYBRID_CANDIDATE_K

    conversation_history = normalize_chat_history(request.history)
    conversation_history_text = format_conversation_history(
        conversation_history
    )
    search_question = build_contextual_search_question(
        question,
        conversation_history,
    )

    if mode == "free_chat":
        system_prompt = (
            "あなたはソード・ワールド2.5の世界観とルールに精通したAIです。"
            "ユーザーの質問には必ずSW2.5の文脈で、具体的かつ専門的に回答してください。"
        )
        llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        messages = [SystemMessage(content=system_prompt)]
        for message in conversation_history:
            if message["role"] == "user":
                messages.append(HumanMessage(content=message["content"]))
            else:
                messages.append(AIMessage(content=message["content"]))
        messages.append(HumanMessage(content=question))
        response = llm.invoke(messages)
        return QueryResponse(
            answer=response.content,
            citations=[],
            sources=[],
            model_used=model_name,
            k_used=0,
            hybrid_k_used=0,
            navigation_pages_used=0,
            structured_pages_used=0,
            reference_pages_used=0,
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
        for citation in citations:
            citation.used_in_answer = True
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
            structured_pages_used=0,
            reference_pages_used=0,
            max_k=0,
            token_usage={},
        )

    # rules_strict
    # 掘り下げ時は直近のユーザー質問も検索文脈へ含める。
    # 過去のAI回答は検索語へ混ぜず、回答生成時の会話理解にだけ利用する。
    variants = build_query_variants(search_question)

    hybrid_items = hybrid_search(
        query=search_question,
        top_k=initial_k,
        candidate_k=max_k,
        books=books,
        variants=variants,
    )
    navigation_pages, _exact_index_match = navigation_search(
        query=search_question,
        books=books,
        variants=variants,
    )
    structured_pages = structured_page_search(
        question=search_question,
        books=books,
        variants=variants,
    )

    context_items, reference_pages = build_context_items(
        question=search_question,
        navigation_pages=navigation_pages,
        structured_pages=structured_pages,
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
            structured_pages_used=0,
            reference_pages_used=0,
            max_k=max_k,
            token_usage={},
        )

    citations = build_citations(context_items, search_question)
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
        source = item.get("source", "")
        source_text = f"\n検索種別: {source}" if source else ""
        context_text = item.get("context_text") or doc.page_content
        context_parts.append(
            f"[CITATION:C{citation_id}]\n"
            f"書籍: {book}\n"
            f"書籍ページ: {logical_page}"
            f"{source_text}"
            f"{reason_text}\n"
            f"本文:\n{context_text}"
        )

    context = "\n\n".join(context_parts)
    full_prompt = prompt.format(
        context=context,
        question=question,
        answer_guidance=build_answer_guidance(search_question),
        conversation_history=conversation_history_text,
    )

    llm = ChatOpenAI(model_name=model_name, temperature=0)
    response = llm.invoke(full_prompt)

    answer = collapse_repeated_citations(response.content)
    returned_citations = select_return_citations(
        answer,
        citations,
        context_items,
        structured_query=query_is_structured(search_question),
        question=search_question,
    )
    sources = citations_to_legacy_sources(returned_citations)

    return QueryResponse(
        answer=answer,
        citations=returned_citations,
        sources=sources,
        model_used=model_name,
        k_used=len(context_items),
        hybrid_k_used=len(hybrid_items),
        navigation_pages_used=len(navigation_pages),
        structured_pages_used=len(structured_pages),
        reference_pages_used=len(reference_pages),
        max_k=max_k,
        token_usage=response.response_metadata.get("token_usage", {}),
    )
