from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import json

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage


# ============================================================
# 環境変数ロード
# ============================================================

load_dotenv()


# ============================================================
# 定数・初期化
# ============================================================

INDEX_DIR = os.getenv(
    "INDEX_DIR",
    "./vector_index",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

embedding = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
)

db = FAISS.load_local(
    INDEX_DIR,
    embedding,
    allow_dangerous_deserialization=True,
)


# ============================================================
# 全文検索・ハイブリッド検索用データ
# ============================================================

# FAISSに格納されている全Documentを保持する。
# exact_search と lexical search で使用する。
all_index_documents = list(
    db.docstore._dict.values()
)

# 日本語では通常の単語単位TF-IDFよりも
# character n-gramの方が固有名詞・戦闘特技名などに強い。
#
# 例:
#   魔力撃
#   魔力
#   力撃
#
# のような文字列一致を検索に利用できる。
lexical_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    min_df=1,
)

lexical_matrix = lexical_vectorizer.fit_transform(
    [
        doc.page_content
        for doc in all_index_documents
    ]
)


# ============================================================
# カテゴリデータ読み込み
# ============================================================

BOOK_CATEGORY_PATH = os.getenv(
    "BOOK_CATEGORY_PATH",
    "book/book_categories.json",
)

with open(
    BOOK_CATEGORY_PATH,
    "r",
    encoding="utf-8",
) as f:
    category_data = json.load(f)


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

        book_name = book_entry["name"]

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
# FastAPI初期化
# ============================================================

app = FastAPI()


# ============================================================
# リクエスト・レスポンスモデル
# ============================================================

class QueryRequest(BaseModel):
    question: str
    books: Optional[List[str]] = None
    model: Optional[str] = "gpt-4.1-nano"
    k: Optional[int] = 10
    mode: Optional[str] = "rules_strict"


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    model_used: Optional[str] = None
    k_used: Optional[int] = None
    max_k: Optional[int] = None
    token_usage: Optional[dict] = None


# ============================================================
# プロンプトテンプレート
# ============================================================

template = """
以下のコンテキストに基づいて、正確に質問に答えてください。

出典を回答に含める場合の注意事項:
- 回答文内で出典を記載する場合は、必ず、コンテキスト中に存在する書籍名とページ番号を参照して、`(書籍名 - p.ページ番号)`の形式で記載してください。
- どの様な場合でも、完全に、`(書籍名 - p.ページ番号)`の形式である必要が有ります。例外は一切ありません。これ以外の形式で回答した場合は、不正解として扱われます。
- `(書籍名 - p.1,p.2)`といった、一つの出典に複数のページをまとめて回答するのは禁止です。`(書籍名 - p.ページ番号1)、(書籍名 - p.ページ番号2)`と回答してください。
- 書籍名やページ番号を勝手に想像して記載してはいけません。コンテキスト中に存在する書籍名は一字一句、完全に使用しなければなりません。
- 出典を囲む記号は、半角括弧()で有る必要が有ります。それ以外の記号は一切認められません。
- 出典は回答文の末尾にまとめず、内容に応じた自然な位置に挿入してください。末尾に列挙するのは、不正解として扱われます。
- これは後続の処理で機械的にリンクを張るために必要なことで、上記条件を満たさない回答は一切行ってはいけません。不具合が発生するため絶対に禁止です。

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
# 共通処理
# ============================================================

def document_key(doc):
    """
    同一Documentを識別するためのキーを生成する。

    新FAISSではchunk番号をmetadataに保持しているため、
    book + page + chunk で一意に識別する。
    """

    return (
        doc.metadata.get("book"),
        doc.metadata.get("page"),
        doc.metadata.get("chunk"),
    )


def get_document_category(doc):
    """
    Documentのカテゴリを取得する。
    metadataにcategoryがない旧データの場合は
    book_categories.jsonから補完する。
    """

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


# ============================================================
# カテゴリ重み付け
# exact_search用
# ============================================================

def apply_category_weight(results):

    boosted_results = []

    for doc in results:

        category = get_document_category(
            doc
        )

        weight = category_weight.get(
            category,
            1.0,
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
        for doc, _ in boosted_results
    ]


# ============================================================
# 日本語character n-gram検索
# ============================================================

def lexical_search(
    query: str,
    top_k: int,
    books=None,
):
    """
    character n-gram TF-IDFによる全文検索。

    ベクトル検索とは独立して全Documentを検索するため、
    FAISS上位候補に入らなかった固有名詞・戦闘特技名なども
    候補に復帰できる。
    """

    query_vector = (
        lexical_vectorizer
        .transform([query])
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
            scores[index]
        )

        # 文字列的関連がまったくない候補以降は不要
        if score <= 0:
            break

        doc = all_index_documents[
            index
        ]

        if (
            books
            and doc.metadata.get("book")
            not in books
        ):
            continue

        results.append(
            (
                doc,
                score,
            )
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# ハイブリッド検索
# ============================================================

def hybrid_search(
    query: str,
    top_k: int,
    candidate_k: int,
    books=None,
):
    """
    FAISSベクトル検索と
    character n-gram lexical検索を統合する。

    スコアの尺度が異なるため、生スコアの加重平均ではなく
    Reciprocal Rank Fusion (RRF) で順位を統合する。

    category weightは検索関連性そのものには使用せず、
    同程度の候補をわずかに優先する補正としてだけ使用する。
    """

    # --------------------------------------------------------
    # ベクトル検索
    # --------------------------------------------------------

    vector_results = (
        db.similarity_search_with_score(
            query,
            k=candidate_k,
        )
    )

    vector_rank = {}

    rank = 0

    for doc, _distance in vector_results:

        if (
            books
            and doc.metadata.get("book")
            not in books
        ):
            continue

        rank += 1

        vector_rank[
            document_key(doc)
        ] = (
            doc,
            rank,
        )


    # --------------------------------------------------------
    # Lexical検索
    # --------------------------------------------------------

    lexical_results = lexical_search(
        query=query,
        top_k=candidate_k,
        books=books,
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
            document_key(doc)
        ] = (
            doc,
            rank,
        )


    # --------------------------------------------------------
    # 候補統合
    # --------------------------------------------------------

    candidate_keys = (
        set(vector_rank.keys())
        |
        set(lexical_rank.keys())
    )

    scored_candidates = []

    # RRFの標準的な安定化定数
    rrf_k = 60.0

    # ベクトルとlexicalの比率。
    # SW2.5では固有名詞一致が重要なのでlexicalをやや優先する。
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

        # --------------------------------------------
        # ベクトル順位
        # --------------------------------------------

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

        # --------------------------------------------
        # Lexical順位
        # --------------------------------------------

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

        # --------------------------------------------
        # カテゴリ補正
        #
        # category weightを直接スコアへ足すと
        # 検索内容よりカテゴリが支配してしまうため、
        # 最大でも数%程度の乗算補正に留める。
        # --------------------------------------------

        category = (
            get_document_category(
                doc
            )
        )

        raw_category_weight = (
            category_weight.get(
                category,
                1.0,
            )
        )

        category_bonus = max(
            0.0,
            raw_category_weight - 1.0,
        )

        score *= (
            1.0
            + 0.05
            * category_bonus
        )

        scored_candidates.append(
            (
                doc,
                score,
            )
        )


    # --------------------------------------------------------
    # 最終順位
    # --------------------------------------------------------

    scored_candidates.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        doc
        for doc, _score
        in scored_candidates[
            :top_k
        ]
    ]


# ============================================================
# メインエンドポイント
# ============================================================

@app.post(
    "/ask",
    response_model=QueryResponse,
)
def ask_question(
    request: QueryRequest,
):

    question = request.question
    books = request.books

    model_name = (
        request.model
        or "gpt-4.1-nano"
    )

    mode = (
        request.mode
        or "rules_strict"
    )

    initial_k = (
        request.k
        or 10
    )

    max_k = 100


    # ========================================================
    # free_chat
    # ========================================================

    if mode == "free_chat":

        system_prompt = (
            "あなたはソード・ワールド2.5の世界観とルールに精通したAIです。"
            "ユーザーの質問には必ずSW2.5の文脈で、具体的かつ専門的に回答してください。"
        )

        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
        )

        messages = [
            SystemMessage(
                content=system_prompt
            ),
            HumanMessage(
                content=question
            ),
        ]

        response = llm.invoke(
            messages
        )

        return QueryResponse(
            answer=response.content,
            sources=[],
            model_used=model_name,
            k_used=0,
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

        for doc in all_index_documents:

            if books and (
                doc.metadata.get("book")
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


        # 同一ページに複数chunkが存在する場合、
        # Discord側へ同一ページを何度も返さないようにする。
        sources = []
        seen_sources = set()

        for doc in results:

            book = doc.metadata.get(
                "book",
                "不明",
            )

            page = doc.metadata.get(
                "page",
                "?",
            )

            source_key = (
                book,
                page,
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(
                source_key
            )

            sources.append(
                f"{book} - p.{page}"
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
            sources=sources,
            model_used="AIは使用していません",
            k_used=0,
            max_k=0,
            token_usage={},
        )


    # ========================================================
    # rules_strict
    # ========================================================

    else:

        selected_docs = hybrid_search(
            query=question,
            top_k=initial_k,
            candidate_k=max_k,
            books=books,
        )


        if not selected_docs:

            return QueryResponse(
                answer=(
                    "該当する情報が"
                    "見つかりませんでした。"
                ),
                sources=[],
                model_used=model_name,
                k_used=0,
                max_k=max_k,
                token_usage={},
            )


        # ----------------------------------------------------
        # LLMへ渡すコンテキスト
        # ----------------------------------------------------

        context = "\n".join(
            (
                f"["
                f"{doc.metadata.get('book', '不明')}"
                f" - p."
                f"{doc.metadata.get('page', '?')}"
                f"]: "
                f"{doc.page_content}"
            )
            for doc
            in selected_docs
        )


        full_prompt = prompt.format(
            context=context,
            question=question,
        )


        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
        )


        response = llm.invoke(
            full_prompt
        )


        # ----------------------------------------------------
        # 出典一覧
        # ----------------------------------------------------

        sources = []

        for doc in selected_docs:

            book = doc.metadata.get(
                "book",
                "不明",
            )

            page = doc.metadata.get(
                "page",
                "?",
            )

            category = (
                get_document_category(
                    doc
                )
            )

            src = (
                f"{category}"
                f" / "
                f"{book}"
                f" - p."
                f"{page}"
            )

            if src not in sources:

                sources.append(
                    src
                )


        return QueryResponse(
            answer=response.content,
            sources=sources,
            model_used=model_name,
            k_used=len(
                selected_docs
            ),
            max_k=max_k,
            token_usage=(
                response
                .response_metadata
                .get(
                    "token_usage",
                    {},
                )
            ),
        )