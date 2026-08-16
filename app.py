import html
import json
import os
import re

import requests
import streamlit as st

from get_page_link import (
    get_citation_label,
    get_citation_links,
)


# ============================================================
# Basic settings
# ============================================================

st.set_page_config(
    page_title="ソード・ワールド2.5 ルールAI bot",
    layout="wide",
)
st.title("📚 ソード・ワールド2.5 ルールAI bot")

API_URL = os.getenv("QA_API_URL", "http://localhost:8000/ask")

if "history" not in st.session_state:
    st.session_state.history = []
if "question_submitted" not in st.session_state:
    st.session_state.question_submitted = False
if "hybrid_k" not in st.session_state:
    st.session_state.hybrid_k = 10


# ============================================================
# Helpers
# ============================================================


def load_book_categories():
    try:
        path = os.getenv(
            "BOOK_CATEGORY_PATH",
            "./book/book_categories.json",
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.sidebar.error(f"カテゴリ情報の読み込みに失敗しました: {exc}")
        return {}


def link_citation_markers(answer: str, citations: list) -> str:
    citation_map = {
        citation.get("id"): citation
        for citation in citations
        if citation.get("id") is not None
    }

    def replacer(match):
        citation_id = int(match.group(1))
        citation = citation_map.get(citation_id)
        if not citation:
            return match.group(0)

        label = get_citation_label(citation)
        pdf_link, image_link = get_citation_links(citation)

        if image_link and pdf_link:
            return (
                f"（<a href='{html.escape(image_link)}' target='_blank'>"
                f"{html.escape(label)}</a> "
                f"[<a href='{html.escape(pdf_link)}' target='_blank'>PDF</a>]）"
            )
        return f"（{html.escape(label)}）"

    return re.sub(r"\[C(\d+)\]", replacer, answer)


model_prices = {
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
}


def calculate_price(model, token_usage):
    price_info = model_prices.get(model, {"input": 0, "output": 0})
    ratio = 150
    input_price = (
        token_usage.get("prompt_tokens", 0)
        / 1_000_000
        * price_info["input"]
        * ratio
    )
    output_price = (
        token_usage.get("completion_tokens", 0)
        / 1_000_000
        * price_info["output"]
        * ratio
    )
    return input_price, output_price, input_price + output_price


def make_history_entry(question: str, result: dict) -> dict:
    return {
        "question": question,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "sources": result.get("sources", []),
        "model_used": result.get("model_used"),
        "token_usage": result.get("token_usage", {}),
        "k_used": result.get("k_used", 0),
        "hybrid_k_used": result.get("hybrid_k_used", 0),
        "navigation_pages_used": result.get("navigation_pages_used", 0),
        "max_k": result.get("max_k", 100),
    }


def render_citation(citation: dict):
    label = get_citation_label(citation)
    pdf_link, image_link = get_citation_links(citation)
    reason = citation.get("reason") or ""
    excerpt = citation.get("excerpt") or ""

    safe_label = html.escape(label)
    safe_reason = html.escape(reason)
    safe_excerpt = html.escape(excerpt)

    image_html = ""
    link_html = safe_label

    if image_link:
        safe_image_link = html.escape(image_link)
        image_html = (
            f"<a href='{safe_image_link}' target='_blank'>"
            f"<img src='{safe_image_link}' "
            f"style='width:110px;border:1px solid #ccc;border-radius:4px;'>"
            f"</a>"
        )
        link_html = (
            f"<a href='{safe_image_link}' target='_blank' "
            f"style='text-decoration:none;font-weight:600;'>{safe_label}</a>"
        )

    pdf_html = ""
    if pdf_link:
        pdf_html = (
            f"<a href='{html.escape(pdf_link)}' target='_blank'>PDFで開く</a>"
        )

    reason_html = ""
    if safe_reason:
        reason_html = (
            "<div style='font-size:0.90em;margin-top:4px;'>"
            "<strong>選定理由:</strong> "
            f"{safe_reason}</div>"
        )

    excerpt_html = ""
    if safe_excerpt:
        excerpt_html = (
            "<div style='font-size:0.90em;margin-top:6px;padding:8px;"
            "background:rgba(127,127,127,0.08);border-radius:4px;'>"
            f"{safe_excerpt}</div>"
        )

    card = f"""
<div style="display:flex;gap:12px;align-items:flex-start;margin:10px 0 16px 0;">
  <div style="flex:0 0 auto;">{image_html}</div>
  <div style="flex:1;min-width:0;">
    <div>{link_html}</div>
    <div style="font-size:0.90em;margin-top:2px;">{pdf_html}</div>
    {reason_html}
    {excerpt_html}
  </div>
</div>
"""
    st.markdown(card, unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================

book_categories = load_book_categories()

book_name_map = {}
for category, category_info in book_categories.items():
    for book_entry in category_info.get("books", []):
        full_name = book_entry["name"]
        display_name = book_entry.get("display_name", full_name)
        book_name_map[display_name] = full_name
        book_name_map[full_name] = full_name

st.sidebar.header("🔍 検索条件")

model_options = {
    "gpt-5.4-nano": "GPT-5.4 Nano (通常)",
    "gpt-5.4-mini": "GPT-5.4 Mini (高性能)",
    "gpt-5.4": "GPT-5.4 (最高性能)",
}

display_model_options = list(model_options.values())
default_model_display = model_options["gpt-5.4-nano"]
selected_model_display = st.sidebar.selectbox(
    "🧠 モデル選択",
    display_model_options,
    index=display_model_options.index(default_model_display),
)
selected_model = next(
    key for key, value in model_options.items() if value == selected_model_display
)

st.sidebar.markdown("📚 **検索対象の書籍**")
selected_books = []

for category, category_info in book_categories.items():
    books = category_info.get("books", [])
    default_checked = category_info.get("default_enabled", True)

    with st.sidebar.expander(category, expanded=True):
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("すべて選択", key=f"select_{category}"):
                for book_entry in books:
                    st.session_state[f"book_{book_entry['name']}"] = True
        with cols[1]:
            if st.button("すべて解除", key=f"clear_{category}"):
                for book_entry in books:
                    st.session_state[f"book_{book_entry['name']}"] = False

        for book_entry in books:
            full_name = book_entry["name"]
            display_name = book_entry.get("display_name", full_name)
            book_key = f"book_{full_name}"
            if book_key not in st.session_state:
                st.session_state[book_key] = default_checked
            if st.checkbox(display_name, key=book_key):
                selected_books.append(full_name)


# ============================================================
# Help
# ============================================================

with st.expander("操作説明 (クリックして開く)"):
    st.markdown(
        """
## このアプリケーションの使い方

1. **質問入力欄**に質問を入力してください。  
2. 回答モードを選択してください。  
3. 必要に応じて検索対象の書籍を選択してください。  
4. 「質問する」をクリックすると、回答と出典が表示されます。  
5. 出典には、書籍ページ・選定理由・該当箇所の抜粋・画像/PDFリンクを表示します。

## 各モード

### ルールブックに基づく回答と出典
- Vector/全文検索に加え、目次・索引を利用して関連ページを検索します。
- 目次・索引で重要と判断されたページは、通常検索の `k` とは別枠で推論コンテキストへ追加されます。
- 回答下部の出典一覧には、AIが回答中で実際に引用したページだけを表示します。

### 全文検索モード
- 入力したキーワードが本文に出現するページを検索します。
- AIは使用しません。
- スペース区切りでAND検索できます。

### AI自由解釈モード
- 保有書籍データを渡さず、AI単体でSW2.5の文脈に沿って回答します。
- 出典・掲載ページの確認には向きません。
"""
    )


# ============================================================
# Question form
# ============================================================

with st.form("question_form"):
    question = st.text_input(
        "質問",
        placeholder="例: マルチアクションはどういった戦闘特技ですか？",
        label_visibility="collapsed",
    )

    mode_display = st.radio(
        "回答モード選択",
        [
            "🛡️ ルールブックに基づく回答と出典",
            "🔍 全文検索モード",
            "💬 AI自由解釈モード",
        ],
        index=0,
    )

    mode_map = {
        "🛡️ ルールブックに基づく回答と出典": "rules_strict",
        "🔍 全文検索モード": "exact_search",
        "💬 AI自由解釈モード": "free_chat",
    }
    selected_mode = mode_map[mode_display]

    submitted = st.form_submit_button("💬 質問する")
    if submitted:
        st.session_state.hybrid_k = 10
        st.session_state.question_submitted = True
        st.session_state.current_question = question
        st.session_state.mode = selected_mode


# ============================================================
# Search execution
# ============================================================

if st.session_state.get("question_submitted"):
    current_question = st.session_state.get("current_question", "")

    if not current_question:
        st.warning("質問を入力してください。")
    else:
        with st.spinner("AIが調査中です..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "question": current_question,
                        "books": selected_books,
                        "model": selected_model,
                        "mode": st.session_state.get("mode", "rules_strict"),
                        "k": st.session_state.get("hybrid_k", 10),
                    },
                    timeout=180,
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.history.append(
                        make_history_entry(current_question, result)
                    )
                    st.session_state.hybrid_k = result.get(
                        "hybrid_k_used",
                        st.session_state.get("hybrid_k", 10),
                    )
                else:
                    st.error(
                        "APIからの応答に失敗しました。"
                        f"コード: {response.status_code}"
                    )
            except Exception as exc:
                st.error(f"エラーが発生しました: {exc}")

    st.session_state.question_submitted = False


# ============================================================
# Expand search
# ============================================================


def expand_search():
    current_k = int(st.session_state.get("hybrid_k", 10))
    new_k = current_k + 10
    max_k = 100

    if new_k > max_k:
        st.warning(f"最大 k 値を超えました: {max_k}")
        return

    st.session_state.hybrid_k = new_k

    try:
        response = requests.post(
            API_URL,
            json={
                "question": st.session_state.get("current_question", ""),
                "books": selected_books,
                "model": selected_model,
                "mode": st.session_state.get("mode", "rules_strict"),
                "k": new_k,
            },
            timeout=180,
        )
    except Exception as exc:
        st.error(f"再質問に失敗しました: {exc}")
        return

    if response.status_code != 200:
        st.error("再質問に失敗しました。")
        return

    result = response.json()
    st.session_state.history.append(
        make_history_entry(
            st.session_state.get("current_question", ""),
            result,
        )
    )
    st.session_state.hybrid_k = result.get("hybrid_k_used", new_k)


# ============================================================
# History display
# ============================================================

for idx, entry in enumerate(reversed(st.session_state.history)):
    with st.chat_message("Q"):
        st.markdown(entry["question"])

    with st.chat_message("A"):
        display_text = link_citation_markers(
            entry["answer"],
            entry.get("citations", []),
        )

        if entry.get("model_used"):
            display_text += f"\n\n🔧 使用モデル: `{entry['model_used']}`"

        if entry.get("token_usage"):
            tokens = entry["token_usage"]
            display_text += (
                "\n\n🧮 トークン数: "
                f"入力 {tokens.get('prompt_tokens', 0)}, "
                f"出力 {tokens.get('completion_tokens', 0)}, "
                f"合計 {tokens.get('total_tokens', 0)}"
            )
            input_price, output_price, total_price = calculate_price(
                entry.get("model_used"),
                tokens,
            )
            display_text += (
                "\n\n💰 推定料金: "
                f"入力: ¥{input_price:.2f} / "
                f"出力: ¥{output_price:.2f} / "
                f"合計: ¥{total_price:.2f}"
            )

        st.markdown(display_text, unsafe_allow_html=True)

        if entry.get("model_used") != "AIは使用していません":
            context_k = int(entry.get("k_used", 0) or 0)
            hybrid_k = int(entry.get("hybrid_k_used", 0) or 0)
            nav_pages = int(entry.get("navigation_pages_used", 0) or 0)
            max_k = int(entry.get("max_k", 100) or 100)

            st.markdown(
                f"📊 推論コンテキスト: **{context_k} chunks** / "
                f"通常検索: **k={hybrid_k}** / "
                f"navigation補完: **{nav_pages} pages**"
            )

            if hybrid_k and hybrid_k < max_k:
                st.button(
                    "通常検索範囲を広げて再質問",
                    key=f"expand_search_{idx}",
                    on_click=expand_search,
                )

        citations = entry.get("citations", [])
        if citations:
            st.markdown("**📖 回答で使用した出典:**")
            for citation in citations:
                try:
                    render_citation(citation)
                except Exception:
                    st.markdown(f"- {citation}")
        elif entry.get("sources"):
            st.markdown("**📖 出典:**")
            for src in entry["sources"]:
                st.markdown(f"- {src}")

        st.markdown("---")


# ============================================================
# Clear history
# ============================================================

st.sidebar.markdown("---")
if st.sidebar.button("🧹 履歴をクリア"):
    st.session_state.history.clear()
    st.rerun()
