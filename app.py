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
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Sidebar chat history: compact, flat rows instead of large framed buttons. */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"] {
    min-height: 1.75rem;
    padding: 0.15rem 0.35rem;
    border: 0;
    background: transparent;
    box-shadow: none;
    justify-content: flex-start;
    text-align: left;
    font-weight: 400;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(127, 127, 127, 0.12);
}
</style>
""",
    unsafe_allow_html=True,
)

API_URL = os.getenv("QA_API_URL", "http://localhost:8000/ask")
API_BASE_URL = API_URL.rsplit("/", 1)[0]
AUTH_ME_URL = API_BASE_URL + "/auth/me"
ADMIN_USERS_URL = API_BASE_URL + "/admin/users"
CHATS_URL = API_BASE_URL + "/chats"
DEFAULT_HYBRID_K = 20

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chat_input_nonce" not in st.session_state:
    st.session_state.chat_input_nonce = 0


# ============================================================
# API helpers
# ============================================================


def get_cloudflare_access_jwt() -> str | None:
    try:
        return st.context.headers.get("Cf-Access-Jwt-Assertion")
    except Exception:
        return None


def get_api_headers() -> dict[str, str]:
    token = get_cloudflare_access_jwt()
    if not token:
        return {}
    return {"Cf-Access-Jwt-Assertion": token}


def _api_error_detail(response: requests.Response, fallback: str) -> str:
    try:
        return response.json().get("detail") or fallback
    except ValueError:
        return fallback


def get_authenticated_user() -> tuple[dict | None, str | None]:
    token = get_cloudflare_access_jwt()
    if not token:
        return None, "Cloudflare Accessの認証情報が見つかりません。"

    try:
        response = requests.get(
            AUTH_ME_URL,
            headers={"Cf-Access-Jwt-Assertion": token},
            timeout=15,
        )
    except requests.RequestException as exc:
        return None, f"認証APIへの接続に失敗しました: {exc}"

    if response.status_code != 200:
        return None, _api_error_detail(response, "認証に失敗しました。")

    return response.json(), None


def get_admin_users() -> tuple[list[dict], str | None]:
    try:
        response = requests.get(
            ADMIN_USERS_URL,
            headers=get_api_headers(),
            timeout=15,
        )
    except requests.RequestException as exc:
        return [], f"ユーザー一覧の取得に失敗しました: {exc}"

    if response.status_code != 200:
        return [], _api_error_detail(
            response,
            "ユーザー一覧の取得に失敗しました。",
        )

    return response.json(), None


def create_admin_user(payload: dict) -> str | None:
    try:
        response = requests.post(
            ADMIN_USERS_URL,
            headers=get_api_headers(),
            json=payload,
            timeout=15,
        )
    except requests.RequestException as exc:
        return f"ユーザー登録に失敗しました: {exc}"

    if response.status_code != 200:
        return _api_error_detail(response, "ユーザー登録に失敗しました。")
    return None


def update_admin_user(payload: dict) -> str | None:
    try:
        response = requests.put(
            ADMIN_USERS_URL,
            headers=get_api_headers(),
            json=payload,
            timeout=15,
        )
    except requests.RequestException as exc:
        return f"ユーザー更新に失敗しました: {exc}"

    if response.status_code != 200:
        return _api_error_detail(response, "ユーザー更新に失敗しました。")
    return None


def get_chats() -> tuple[list[dict], str | None]:
    try:
        response = requests.get(
            CHATS_URL,
            headers=get_api_headers(),
            timeout=15,
        )
    except requests.RequestException as exc:
        return [], f"チャット一覧の取得に失敗しました: {exc}"

    if response.status_code != 200:
        return [], _api_error_detail(
            response,
            "チャット一覧の取得に失敗しました。",
        )

    return response.json(), None


def create_chat() -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            CHATS_URL,
            headers=get_api_headers(),
            json={"title": ""},
            timeout=15,
        )
    except requests.RequestException as exc:
        return None, f"チャットの作成に失敗しました: {exc}"

    if response.status_code != 200:
        return None, _api_error_detail(
            response,
            "チャットの作成に失敗しました。",
        )

    return response.json(), None


def get_chat_detail(chat_id: str) -> tuple[dict | None, str | None]:
    try:
        response = requests.get(
            f"{CHATS_URL}/{chat_id}",
            headers=get_api_headers(),
            timeout=15,
        )
    except requests.RequestException as exc:
        return None, f"チャットの取得に失敗しました: {exc}"

    if response.status_code != 200:
        return None, _api_error_detail(
            response,
            "チャットの取得に失敗しました。",
        )

    return response.json(), None


def delete_chat(chat_id: str) -> str | None:
    try:
        response = requests.delete(
            f"{CHATS_URL}/{chat_id}",
            headers=get_api_headers(),
            timeout=15,
        )
    except requests.RequestException as exc:
        return f"チャットの削除に失敗しました: {exc}"

    if response.status_code != 200:
        return _api_error_detail(
            response,
            "チャットの削除に失敗しました。",
        )

    return None


def ask_question(
    question: str,
    *,
    chat_id: str,
    books: list[str],
    model: str,
    mode: str,
) -> str | None:
    try:
        response = requests.post(
            API_URL,
            headers=get_api_headers(),
            json={
                "question": question,
                "books": books,
                "model": model,
                "mode": mode,
                "k": DEFAULT_HYBRID_K,
                "chat_id": chat_id,
            },
            timeout=180,
        )
    except requests.RequestException as exc:
        return f"質問APIへの接続に失敗しました: {exc}"

    if response.status_code != 200:
        return _api_error_detail(
            response,
            f"APIからの応答に失敗しました。コード: {response.status_code}",
        )

    return None


# ============================================================
# Display helpers
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


book_display_name_map = {}


def link_citation_markers(answer: str, citations: list) -> str:
    if not answer:
        return answer

    citation_map = {}
    for citation in citations:
        citation_id = citation.get("id")
        if citation_id is None:
            continue
        try:
            citation_id = int(citation_id)
        except (TypeError, ValueError):
            continue
        citation_map.setdefault(citation_id, citation)

    def replacer(match):
        citation_id = int(match.group(1))
        citation = citation_map.get(citation_id)
        if not citation:
            return match.group(0)

        source_type = citation.get("source_type") or "book"

        if source_type == "official_correction":
            label = "[公式エラッタ]"
            target_link = citation.get("source_url") or ""
        else:
            book = citation.get("book") or ""
            display_name = book_display_name_map.get(
                book,
                book or f"C{citation_id}",
            )
            page = citation.get("page")

            if page is not None:
                label = f"[{display_name} p.{page}]"
            else:
                label = f"[{display_name}]"

            pdf_link, image_link = get_citation_links(citation)
            target_link = pdf_link or image_link

        safe_label = html.escape(label)
        if not target_link:
            return safe_label

        safe_link = html.escape(target_link, quote=True)
        return (
            f"<a href='{safe_link}' "
            f"target='_blank' "
            f"title='出典を開く'>"
            f"{safe_label}"
            f"</a>"
        )

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


def render_citation(citation: dict):
    source_type = citation.get("source_type") or "book"
    reason = citation.get("reason") or ""
    excerpt = citation.get("excerpt") or ""
    used_in_answer = citation.get("used_in_answer", True)

    if used_in_answer:
        badge_html = (
            "<span style='display:inline-block;"
            "padding:2px 7px;"
            "margin-left:6px;"
            "font-size:0.78em;"
            "border-radius:10px;"
            "background:rgba(46,160,67,0.15);'>"
            "回答で引用"
            "</span>"
        )
    else:
        badge_html = (
            "<span style='display:inline-block;"
            "padding:2px 7px;"
            "margin-left:6px;"
            "font-size:0.78em;"
            "border-radius:10px;"
            "background:rgba(31,111,235,0.12);'>"
            "関連資料"
            "</span>"
        )

    safe_reason = html.escape(reason)
    safe_excerpt = html.escape(excerpt)

    reason_html = ""
    if safe_reason:
        reason_html = (
            "<div style='font-size:0.90em;"
            "margin-top:4px;'>"
            "<strong>選定理由:</strong> "
            f"{safe_reason}</div>"
        )

    excerpt_html = ""
    if safe_excerpt:
        excerpt_html = (
            "<div style='font-size:0.90em;"
            "margin-top:6px;"
            "padding:8px;"
            "background:rgba(127,127,127,0.08);"
            "border-radius:4px;'>"
            f"{safe_excerpt}</div>"
        )

    if source_type == "official_correction":
        source_name = (
            citation.get("book")
            or "GroupSNE ソード・ワールド2.5 エラッタ・追加データ"
        )
        source_url = citation.get("source_url") or ""
        target_book = citation.get("target_book") or ""
        target_page = citation.get("target_page")
        operation = citation.get("operation") or ""

        safe_source_name = html.escape(source_name)

        if source_url:
            safe_source_url = html.escape(source_url, quote=True)
            link_html = (
                f"<a href='{safe_source_url}' "
                f"target='_blank' "
                f"style='text-decoration:none;"
                f"font-weight:600;'>"
                f"{safe_source_name}</a>"
            )
        else:
            link_html = (
                f"<span style='font-weight:600;'>"
                f"{safe_source_name}</span>"
            )

        detail_parts = ["公式エラッタ"]

        if target_book:
            target_display = book_display_name_map.get(
                target_book,
                target_book,
            )
            if target_page is not None:
                detail_parts.append(
                    f"対象: {target_display} p.{target_page}"
                )
            else:
                detail_parts.append(
                    f"対象: {target_display}"
                )

        operation_labels = {
            "replace": "置換",
            "delete": "削除",
            "append": "追記",
        }
        if operation:
            detail_parts.append(
                "訂正種別: "
                + operation_labels.get(operation, operation)
            )

        detail_html = (
            "<div style='font-size:0.88em;"
            "margin-top:4px;"
            "opacity:0.82;'>"
            + html.escape(" / ".join(detail_parts))
            + "</div>"
        )

        card = f"""
<div style="display:flex;gap:12px;align-items:flex-start;margin:10px 0 16px 0;">
  <div style="flex:1;min-width:0;">
    <div>{link_html}{badge_html}</div>
    {detail_html}
    {reason_html}
    {excerpt_html}
  </div>
</div>
"""
        st.markdown(card, unsafe_allow_html=True)
        return

    label = get_citation_label(citation)
    pdf_link, image_link = get_citation_links(citation)

    safe_label = html.escape(label)
    image_html = ""
    link_html = safe_label

    if image_link:
        safe_image_link = html.escape(
            image_link,
            quote=True,
        )
        image_html = (
            f"<a href='{safe_image_link}' "
            f"target='_blank'>"
            f"<img src='{safe_image_link}' "
            f"style='width:110px;"
            f"border:1px solid #ccc;"
            f"border-radius:4px;'>"
            f"</a>"
        )

    if pdf_link:
        safe_pdf_link = html.escape(
            pdf_link,
            quote=True,
        )
        link_html = (
            f"<a href='{safe_pdf_link}' "
            f"target='_blank' "
            f"style='text-decoration:none;"
            f"font-weight:600;'>"
            f"{safe_label}</a>"
        )
    elif image_link:
        link_html = (
            f"<a href='{safe_image_link}' "
            f"target='_blank' "
            f"style='text-decoration:none;"
            f"font-weight:600;'>"
            f"{safe_label}</a>"
        )

    card = f"""
<div style="display:flex;gap:12px;align-items:flex-start;margin:10px 0 16px 0;">
  <div style="flex:0 0 auto;">{image_html}</div>
  <div style="flex:1;min-width:0;">
    <div>{link_html}{badge_html}</div>
    {reason_html}
    {excerpt_html}
  </div>
</div>
"""
    st.markdown(card, unsafe_allow_html=True)


def render_assistant_message(message: dict):
    metadata = message.get("metadata") or {}
    citations = metadata.get("citations", [])

    display_text = link_citation_markers(
        message.get("content", ""),
        citations,
    )

    # 回答本文は常に表示する。
    st.markdown(display_text, unsafe_allow_html=True)

    technical_lines = []

    if metadata.get("model_used"):
        technical_lines.append(
            f"🔧 使用モデル: `{metadata['model_used']}`"
        )

    if metadata.get("token_usage"):
        tokens = metadata["token_usage"]
        technical_lines.append(
            "🧮 トークン数: "
            f"入力 {tokens.get('prompt_tokens', 0)}, "
            f"出力 {tokens.get('completion_tokens', 0)}, "
            f"合計 {tokens.get('total_tokens', 0)}"
        )

        input_price, output_price, total_price = calculate_price(
            metadata.get("model_used"),
            tokens,
        )
        technical_lines.append(
            "💰 推定料金: "
            f"入力: ¥{input_price:.2f} / "
            f"出力: ¥{output_price:.2f} / "
            f"合計: ¥{total_price:.2f}"
        )

    if metadata.get("model_used") != "AIは使用していません":
        context_k = int(metadata.get("k_used", 0) or 0)
        hybrid_k = int(metadata.get("hybrid_k_used", 0) or 0)
        nav_pages = int(metadata.get("navigation_pages_used", 0) or 0)
        structured_pages = int(
            metadata.get("structured_pages_used", 0) or 0
        )
        reference_pages = int(
            metadata.get("reference_pages_used", 0) or 0
        )
        technical_lines.append(
            f"📊 推論コンテキスト: **{context_k} chunks** / "
            f"通常検索: **k={hybrid_k}** / "
            f"navigation: **{nav_pages} pages** / "
            f"表・一覧: **{structured_pages} pages** / "
            f"参照先: **{reference_pages} pages**"
        )

    if technical_lines:
        with st.expander("技術情報", expanded=False):
            st.markdown("\n\n".join(technical_lines))

    if citations:
        used_citations = [
            citation
            for citation in citations
            if citation.get("used_in_answer", True)
        ]
        related_citations = [
            citation
            for citation in citations
            if not citation.get("used_in_answer", True)
        ]

        if used_citations:
            with st.expander(
                f"📖 回答で使用した出典 ({len(used_citations)})",
                expanded=False,
            ):
                for citation in used_citations:
                    render_citation(citation)

        if related_citations:
            with st.expander(
                f"🔎 関連資料 ({len(related_citations)})",
                expanded=False,
            ):
                st.caption(
                    "回答本文では直接引用していませんが、"
                    "確認価値が高い関連ページです。"
                )
                for citation in related_citations:
                    render_citation(citation)

    elif metadata.get("sources"):
        with st.expander("📖 出典", expanded=False):
            for source in metadata["sources"]:
                st.markdown(f"- {source}")


# ============================================================
# Authentication
# ============================================================

authenticated_user, authentication_error = get_authenticated_user()

if not authenticated_user:
    st.sidebar.error("利用者を確認できませんでした。")
    st.error(
        authentication_error
        or "Cloudflare Accessの認証情報を確認できません。"
    )
    st.stop()

display_name = (
    authenticated_user.get("display_name")
    or authenticated_user["email"]
)


# ============================================================
# Sidebar: product / user
# ============================================================

st.sidebar.title("📚 SW2.5 ルールAI bot")
st.sidebar.caption(display_name)
st.sidebar.caption(authenticated_user["email"])


# ============================================================
# Sidebar: chats
# ============================================================

st.sidebar.subheader("💬 チャット")

if st.sidebar.button(
    "＋ 新しいチャット",
    use_container_width=True,
    type="primary",
):
    # ここではDBへ保存しない。
    # 最初の質問を送信した時点で初めてチャットを作成する。
    st.session_state.current_chat_id = None
    st.session_state.chat_input_nonce += 1
    st.rerun()

chat_list, chat_list_error = get_chats()

if chat_list_error:
    st.sidebar.error(chat_list_error)
    chat_list = []

valid_chat_ids = {chat["id"] for chat in chat_list}

# 削除済みなどで無効なIDだけ新規状態へ戻す。
# current_chat_id=None は「まだ保存されていない新しいチャット」として維持する。
if (
    st.session_state.current_chat_id is not None
    and st.session_state.current_chat_id not in valid_chat_ids
):
    st.session_state.current_chat_id = None
    st.session_state.chat_input_nonce += 1

for chat in chat_list:
    title = chat.get("title") or "新しいチャット"
    active = chat["id"] == st.session_state.current_chat_id

    if st.sidebar.button(
        ("› " if active else "") + title,
        key=f"chat_select_{chat['id']}",
        use_container_width=True,
    ):
        if not active:
            st.session_state.current_chat_id = chat["id"]
            # チャット切替時は、入力途中の文字列を引き継がない。
            st.session_state.chat_input_nonce += 1
            st.rerun()

# ============================================================
# Sidebar: search settings
# ============================================================

book_categories = load_book_categories()

book_display_name_map.clear()
for category, category_info in book_categories.items():
    for book_entry in category_info.get("books", []):
        full_name = book_entry["name"]
        display_name_for_book = book_entry.get(
            "display_name",
            full_name,
        )
        book_display_name_map[full_name] = display_name_for_book

st.sidebar.divider()
st.sidebar.subheader("🔍 検索条件")

model_options = {
    "gpt-5.4-nano": "GPT-5.4 Nano (通常)",
    "gpt-5.4-mini": "GPT-5.4 Mini (高性能)",
    "gpt-5.4": "GPT-5.4 (最高性能)",
}

display_model_options = list(model_options.values())
default_model_display = model_options["gpt-5.4-nano"]

selected_model_display = st.sidebar.selectbox(
    "🧠 モデル",
    display_model_options,
    index=display_model_options.index(default_model_display),
)
selected_model = next(
    key
    for key, value in model_options.items()
    if value == selected_model_display
)

mode_display = st.sidebar.radio(
    "回答モード",
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

st.sidebar.markdown("📚 **検索対象の書籍**")
selected_books = []

for category, category_info in book_categories.items():
    books = category_info.get("books", [])
    default_checked = category_info.get("default_enabled", True)

    with st.sidebar.expander(category, expanded=False):
        cols = st.columns([1, 1])

        with cols[0]:
            if st.button(
                "すべて選択",
                key=f"select_{category}",
                use_container_width=True,
            ):
                for book_entry in books:
                    st.session_state[
                        f"book_{book_entry['name']}"
                    ] = True
                st.rerun()

        with cols[1]:
            if st.button(
                "すべて解除",
                key=f"clear_{category}",
                use_container_width=True,
            ):
                for book_entry in books:
                    st.session_state[
                        f"book_{book_entry['name']}"
                    ] = False
                st.rerun()

        for book_entry in books:
            full_name = book_entry["name"]
            display_name_for_book = book_entry.get(
                "display_name",
                full_name,
            )
            book_key = f"book_{full_name}"

            if book_key not in st.session_state:
                st.session_state[book_key] = default_checked

            if st.checkbox(
                display_name_for_book,
                key=book_key,
            ):
                selected_books.append(full_name)


# ============================================================
# Sidebar: help
# ============================================================

st.sidebar.divider()

with st.sidebar.expander("ℹ️ 操作説明", expanded=False):
    st.markdown(
        """
### 基本操作

- 下部の入力欄から質問してください。
- 同じチャットで質問を続けると、過去の会話と参照資料を踏まえて回答します。
- 別テーマとして分けたい場合は「＋ 新しいチャット」を使用してください。
- 左側のチャット履歴を選ぶと、以前の調査を再開できます。

### ルールブックに基づく回答と出典

- Vector/全文検索に加え、目次・索引を利用して関連ページを検索します。
- 本文中の「○頁参照」「次頁」などの参照先も追跡します。
- 表・テーブル・一覧の質問では、表本体も追加探索します。
- 同一チャットでは、会話履歴と過去に参照した資料を保持します。

### 全文検索モード

- 入力したキーワードが本文に出現するページを検索します。
- AIは使用しません。
- スペース区切りでAND検索できます。

### AI自由解釈モード

- 保有書籍データを渡さず、AI単体で回答します。
- 出典や掲載ページの確認には向きません。
"""
    )


# ============================================================
# Sidebar: admin
# ============================================================

if authenticated_user.get("is_admin"):
    with st.sidebar.expander("⚙️ ユーザー管理", expanded=False):
        admin_users, admin_users_error = get_admin_users()

        if admin_users_error:
            st.error(admin_users_error)
        else:
            st.caption(f"登録ユーザー: {len(admin_users)}人")
            st.caption(
                "Cloudflare Accessで認証された未登録ユーザーは、"
                "初回アクセス時に自動登録されます。"
            )

            with st.form(
                "admin_add_user_form",
                clear_on_submit=True,
            ):
                st.markdown("**ユーザー追加**")
                add_email = st.text_input(
                    "メールアドレス",
                    key="admin_add_email",
                )
                add_display_name = st.text_input(
                    "表示名",
                    key="admin_add_display_name",
                )
                add_is_admin = st.checkbox(
                    "管理者",
                    value=False,
                    key="admin_add_is_admin",
                )
                add_submitted = st.form_submit_button(
                    "追加",
                    use_container_width=True,
                )

            if add_submitted:
                error = create_admin_user(
                    {
                        "email": add_email,
                        "display_name": add_display_name,
                        "is_admin": add_is_admin,
                    }
                )

                if error:
                    st.error(error)
                else:
                    st.success("ユーザーを追加しました。")
                    st.rerun()

            st.divider()
            st.markdown("**登録済みユーザーの編集**")

            if admin_users:
                user_labels = {
                    (
                        f"{user.get('display_name') or user['email']} "
                        f"<{user['email']}>"
                    ): user
                    for user in admin_users
                }

                selected_label = st.selectbox(
                    "対象ユーザー",
                    list(user_labels.keys()),
                    key="admin_edit_user_select",
                )
                selected_user = user_labels[selected_label]
                selected_email = selected_user["email"]

                is_self = (
                    selected_email.strip().lower()
                    == authenticated_user["email"].strip().lower()
                )

                with st.form(
                    f"admin_edit_user_form_{selected_email}"
                ):
                    edit_email = st.text_input(
                        "メールアドレス",
                        value=selected_user["email"],
                        disabled=is_self,
                    )
                    edit_display_name = st.text_input(
                        "表示名",
                        value=(
                            selected_user.get("display_name")
                            or ""
                        ),
                    )
                    edit_is_admin = st.checkbox(
                        "管理者",
                        value=bool(
                            selected_user.get("is_admin")
                        ),
                        disabled=is_self,
                    )

                    if selected_user.get("last_seen_at"):
                        st.caption(
                            "最終アクセス: "
                            + selected_user["last_seen_at"]
                        )

                    if is_self:
                        st.caption(
                            "ログイン中の管理者自身は、"
                            "メール変更・管理者解除できません。"
                        )

                    edit_submitted = st.form_submit_button(
                        "変更を保存",
                        use_container_width=True,
                    )

                if edit_submitted:
                    error = update_admin_user(
                        {
                            "current_email": (
                                selected_user["email"]
                            ),
                            "email": (
                                selected_user["email"]
                                if is_self
                                else edit_email
                            ),
                            "display_name": edit_display_name,
                            "is_admin": (
                                bool(
                                    selected_user.get(
                                        "is_admin"
                                    )
                                )
                                if is_self
                                else edit_is_admin
                            ),
                        }
                    )

                    if error:
                        st.error(error)
                    else:
                        st.success(
                            "ユーザー情報を更新しました。"
                        )
                        st.rerun()


# ============================================================
# Main chat
# ============================================================

if st.session_state.current_chat_id:
    toolbar_left, toolbar_right = st.columns([12, 1])
    with toolbar_right:
        if st.button(
            "🗑️",
            key="delete_current_chat_main",
            help="このチャットを削除",
        ):
            error = delete_chat(st.session_state.current_chat_id)
            if error:
                st.error(error)
            else:
                st.session_state.current_chat_id = None
                st.session_state.chat_input_nonce += 1
                st.rerun()

    current_chat, current_chat_error = get_chat_detail(
        st.session_state.current_chat_id
    )

    if current_chat_error:
        st.error(current_chat_error)
        st.stop()

    messages = current_chat.get("messages", [])
else:
    # 未保存の新規チャット。
    # 最初の質問を送るまではDBにレコードを作らない。
    current_chat = None
    messages = []

# ChatGPT同様、メイン画面には会話だけを上から時系列に並べる。
# 新規チャットでは何も表示せず、下部の入力欄だけになる。
for message in messages:
    role = message.get("role")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(message.get("content", ""))
        continue

    if role == "assistant":
        with st.chat_message("assistant"):
            render_assistant_message(message)


# ============================================================
# Chat input
# ============================================================

input_chat_key = (
    st.session_state.current_chat_id
    if st.session_state.current_chat_id
    else "new"
)

question = st.chat_input(
    "質問を入力してください",
    key=(
        f"chat_input_"
        f"{input_chat_key}_"
        f"{st.session_state.chat_input_nonce}"
    ),
)

if question:
    # 応答待ちの間も、送信した質問が画面上で見えるようにする。
    with st.chat_message("user"):
        st.markdown(question)

    created_chat_id = None
    target_chat_id = st.session_state.current_chat_id

    # 未保存の新規チャットなら、最初の質問を送信したこの時点で初めて作成する。
    if not target_chat_id:
        new_chat, create_error = create_chat()

        if create_error:
            st.error(create_error)
        elif new_chat:
            created_chat_id = new_chat["id"]
            target_chat_id = created_chat_id
            st.session_state.current_chat_id = target_chat_id

    if target_chat_id:
        with st.chat_message("assistant"):
            with st.spinner("AIが調査中です..."):
                error = ask_question(
                    question,
                    chat_id=target_chat_id,
                    books=selected_books,
                    model=selected_model,
                    mode=selected_mode,
                )

            if error:
                # 今回の送信のためだけに作ったチャットで失敗した場合は、
                # 空チャットを履歴に残さない。
                if created_chat_id:
                    delete_chat(created_chat_id)
                    st.session_state.current_chat_id = None
                st.error(error)
            else:
                # 回答取得後は必ず新しい空の入力ウィジェットへ切り替える。
                st.session_state.chat_input_nonce += 1
                st.rerun()
