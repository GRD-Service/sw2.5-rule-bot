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

# --- 環境変数ロード ---
load_dotenv()

# --- 定数・初期化 ---
INDEX_DIR = os.getenv("INDEX_DIR", "./vector_index")
embedding = OpenAIEmbeddings()
db = FAISS.load_local(INDEX_DIR, embedding, allow_dangerous_deserialization=True)

# --- カテゴリデータ読み込み ---
with open(os.getenv("BOOK_CATEGORY_PATH", "book/book_categories.json"), "r", encoding="utf-8") as f:
    category_data = json.load(f)

book_to_category = {}
category_weight = {}

sorted_categories = sorted(category_data.items(), key=lambda item: item[1].get('weight', 1.0), reverse=True)
for category, info in sorted_categories:
    books_sorted = sorted(info["books"], key=lambda x: x["name"])
    for book_entry in books_sorted:
        book_name = book_entry["name"]
        book_to_category[book_name] = category
    category_weight[category] = info.get("weight", 1.0)

# --- FastAPI初期化 ---
app = FastAPI()

# --- リクエスト・レスポンスモデル ---
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

# --- プロンプトテンプレート ---
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
    input_variables=["context", "question"],
    template=template,
)

# --- カテゴリ重み付け関数 ---
def apply_category_weight(results):
    boosted_results = []
    for doc in results:
        book = doc.metadata.get("book")
        category = doc.metadata.get("category", book_to_category.get(book, "その他"))
        weight = category_weight.get(category, 1.0)
        boosted_results.append((doc, weight))
    boosted_results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in boosted_results]

# --- TF-IDFによるスコアリング選別 ---
def select_chunks_by_tfidf(query: str, documents, top_k: int):
    texts = [doc.page_content for doc in documents]
    vectorizer = TfidfVectorizer().fit([query] + texts)
    tfidf_matrix = vectorizer.transform([query] + texts)
    query_vec = tfidf_matrix[0:1]
    doc_vecs = tfidf_matrix[1:]
    scores = cosine_similarity(query_vec, doc_vecs).flatten()
    sorted_indices = scores.argsort()[::-1][:top_k]
    return [documents[i] for i in sorted_indices]

# --- 質問キーワード抽出 ---
# （現在は無効化中 - 分割なし）
def extract_keywords(text: str) -> str:
    return text

# --- メインエンドポイント ---
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    question = request.question
    books = request.books
    model_name = request.model or "gpt-4.1-nano"
    mode = request.mode or "rules_strict"
    initial_k = request.k or 10
    max_k = 100

    if mode == "free_chat":
        system_prompt = (
            "あなたはソード・ワールド2.5の世界観とルールに精通したAIです。"
            "ユーザーの質問には必ずSW2.5の文脈で、具体的かつ専門的に回答してください。"
        )
        llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        response = llm.invoke(messages)
        return QueryResponse(
            answer=response.content,
            sources=[],
            model_used=model_name,
            k_used=0,
            max_k=0,
            token_usage=response.response_metadata.get('token_usage', {}),
        )

    elif mode == "exact_search":
        keywords = question.strip().split()
        results = []
        for doc in db.docstore._dict.values():
            if all(keyword in doc.page_content for keyword in keywords) and (not books or doc.metadata.get('book') in books):
                results.append(doc)
        results = apply_category_weight(results)
        sources = [f"{doc.metadata.get('book', '不明')} - p.{doc.metadata.get('page', '?')}" for doc in results]
        answer = "全文検索を実施しました。結果は出典に記載されています。" if sources else "該当はありませんでした。"
        return QueryResponse(
            answer=answer,
            sources=sources,
            model_used="AIは使用していません",
            k_used=0,
            max_k=0,
            token_usage={},
        )

    else:  # rules_strict
        all_docs = db.docstore._dict.values()
        if books:
            all_docs = [doc for doc in all_docs if doc.metadata.get("book") in books]

        search_query = question
        all_candidates = db.similarity_search(search_query, k=max_k)
        filtered_candidates = [doc for doc in all_candidates if not books or doc.metadata.get("book") in books]
        candidates = apply_category_weight(filtered_candidates)
        def vector_scores_by_query(query: str, documents):
            texts = [doc.page_content for doc in documents]
            embeddings = embedding.embed_documents(texts)
            query_vec = embedding.embed_query(query)
            return [float(cosine_similarity([query_vec], [vec])[0][0]) for vec in embeddings]

        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [doc.page_content for doc in candidates]
        vectorizer = TfidfVectorizer().fit([search_query] + texts)
        tfidf_matrix = vectorizer.transform([search_query] + texts)
        tfidf_query = tfidf_matrix[0:1]
        tfidf_docs = tfidf_matrix[1:]
        tfidf_scores = cosine_similarity(tfidf_query, tfidf_docs).flatten()

        vec_scores = vector_scores_by_query(search_query, candidates)
        category_weights = [category_weight.get(doc.metadata.get("category", "その他"), 1.0) for doc in candidates]
        combined_scores = [
            (i, 0.2 * tfidf_scores[i] + 0.3 * vec_scores[i] + 0.5 * category_weights[i])
            for i in range(len(candidates))
        ]
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        selected_docs = [candidates[i] for i, _ in combined_scores[:initial_k]]

        if not selected_docs:
            return QueryResponse(
                answer="該当する情報が見つかりませんでした。",
                sources=[],
                model_used=model_name,
                k_used=0,
                max_k=max_k,
                token_usage={},
            )

        context = "\n".join(
            f"[{doc.metadata.get('book', '不明')} - p.{doc.metadata.get('page', '?')}]: {doc.page_content}"
            for doc in selected_docs
        )
        full_prompt = prompt.format(context=context, question=question)
        llm = ChatOpenAI(model_name=model_name, temperature=0)
        response = llm.invoke(full_prompt)

        sources = []
        for doc in selected_docs:
            book = doc.metadata.get("book", "不明")
            page = doc.metadata.get("page", "?")
            category = doc.metadata.get("category", "不明")
            src = f"{category} / {book} - p.{page}"
            if src not in sources:
                sources.append(src)

        return QueryResponse(
            answer=response.content,
            sources=sources,
            model_used=model_name,
            k_used=len(selected_docs),
            max_k=max_k,
            token_usage=response.response_metadata.get('token_usage', {}),
        )
