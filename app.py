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
API_BASE_URL = API_URL.rsplit("/", 1)[0]
AUTH_ME_URL = API_BASE_URL + "/auth/me"
ADMIN_USERS_URL = API_BASE_URL + "/admin/users"
CHATS_URL = API_BASE_URL + "/chats"
DEFAULT_HYBRID_K = 20


def get_cloudflare_access_jwt() -> str | None:
    try:
        return st.context.headers.get("Cf-Access-Jwt-Assertion")
    except Exception:
        return None


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
        detail = "認証に失敗しました。"
        try:
            detail = response.json().get("detail") or detail
        except ValueError:
            pass
        return None, detail

    return response.json(), None


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
        return [], _api_error_detail(response, "ユーザー一覧の取得に失敗しました。")
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
        response = requests.get(CHATS_URL, headers=get_api_headers(), timeout=15)
    except requests.RequestException as exc:
        return [], f"チャット一覧の取得に失敗しました: {exc}"
    if response.status_code != 200:
        return [], _api_error_detail(response, "チャット一覧の取得に失敗しました。")
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
        return None, _api_error_detail(response, "チャットの作成に失敗しました。")
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
        return None, _api_error_detail(response, "チャットの取得に失敗しました。")
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
        return _api_error_detail(response, "チャットの削除に失敗しました。")
    return None


authenticated_user, authentication_error = get_authenticated_user()

if not authenticated_user:
    st.sidebar.error("利用者を確認できませんでした。")
    st.error(authentication_error or "このアカウントでは利用できません。")
    st.stop()

display_name = authenticated_user.get("display_name") or authenticated_user["email"]
st.sidebar.success(
    f"認証ユーザー: {display_name}\n\n{authenticated_user['email']}"
)

if authenticated_user.get("is_admin"):
    with st.sidebar.expander("⚙️ ユーザー管理", expanded=False):
        admin_users, admin_users_error = get_admin_users()

        if admin_users_error:
            st.error(admin_users_error)
        else:
            st.caption(f"登録ユーザー: {len(admin_users)}人")
            st.caption("Cloudflare Accessで認証された未登録ユーザーは初回アクセス時に自動登録されます。")

            with st.form("admin_add_user_form", clear_on_submit=True):
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

                with st.form(f"admin_edit_user_form_{selected_email}"):
                    edit_email = st.text_input(
                        "メールアドレス",
                        value=selected_user["email"],
                        disabled=is_self,
                    )
                    edit_display_name = st.text_input(
                        "表示名",
                        value=selected_user.get("display_name") or "",
                    )
                    edit_is_admin = st.checkbox(
                        "管理者",
                        value=bool(selected_user.get("is_admin")),
                        disabled=is_self,
                    )
                    if selected_user.get("last_seen_at"):
                        st.caption(
                            "最終アクセス: "
                            + selected_user["last_seen_at"]
                        )
                    if is_self:
                        st.caption(
                            "ログイン中の管理者自身は、メール変更・"
                            "管理者解除できません。"
                        )

                    edit_submitted = st.form_submit_button(
                        "変更を保存",
                        use_container_width=True,
                    )

                if edit_submitted:
                    error = update_admin_user(
                        {
                            "current_email": selected_user["email"],
                            "email": (
                                selected_user["email"] if is_self else edit_email
                            ),
                            "display_name": edit_display_name,
                            "is_admin": (
                                bool(selected_user.get("is_admin"))
                                if is_self
                                else edit_is_admin
                            ),
                        }
                    )
                    if error:
                        st.error(error)
                    else:
                        st.success("ユーザー情報を更新しました。")
                        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("💬 チャット")

if st.sidebar.button("＋ 新しいチャット", use_container_width=True):
    new_chat, error = create_chat()
    if error:
        st.sidebar.error(error)
    elif new_chat:
        st.session_state.current_chat_id = new_chat["id"]
        st.rerun()

chat_list, chat_list_error = get_chats()
if chat_list_error:
    st.sidebar.error(chat_list_error)
    chat_list = []

valid_chat_ids = {chat["id"] for chat in chat_list}
if st.session_state.get("current_chat_id") not in valid_chat_ids:
    st.session_state.current_chat_id = chat_list[0]["id"] if chat_list else None

if not st.session_state.get("current_chat_id"):
    new_chat, error = create_chat()
    if error:
        st.error(error)
        st.stop()
    st.session_state.current_chat_id = new_chat["id"]
    st.rerun()

for chat in chat_list:
    title = chat.get("title") or "新しいチャット"
    active = chat["id"] == st.session_state.current_chat_id
    label = ("● " if active else "") + title
    if st.sidebar.button(
        label,
        key=f"chat_select_{chat['id']}",
        use_container_width=True,
    ):
        st.session_state.current_chat_id = chat["id"]
        st.rerun()

current_chat, current_chat_error = get_chat_detail(
    st.session_state.current_chat_id
)
if current_chat_error:
    st.error(current_chat_error)
    st.stop()

with st.sidebar.expander("現在のチャット", expanded=False):
    st.caption(current_chat.get("title") or "新しいチャット")
    if st.button(
        "このチャットを削除",
        key="delete_current_chat",
        use_container_width=True,
    ):
        error = delete_chat(st.session_state.current_chat_id)
        if error:
            st.error(error)
        else:
            st.session_state.current_chat_id = None
            st.rerun()


if "question_submitted" not in st.session_state:
    st.session_state.question_submitted = False
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None


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


# 正式書名 -> UI表示名。book_categories.json 読み込み後に構築する。
book_display_name_map = {}


def link_citation_markers(answer: str, citations: list) -> str:
    """
    回答中の [C5] 等を、book_categories.json の display_name と
    書籍ページを使った短い表示へ変換する。

    例:
      [C5]  -> [ルールブック1 p.292]
      [C13] -> [バトルマスタリー p.37]

    - API/CLI内部の Citation ID は変更しない。
    - GUI表示時だけ display_name + 論理ページへ変換する。
    - 表示文字列全体を画像ページへのリンクにする。
    - 画像リンクがなければ PDF リンクへフォールバックする。
    - citations に同一 ID が重複していても最初の1件だけを採用する。
    """
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

        book = citation.get("book") or ""
        display_name = book_display_name_map.get(book, book or f"C{citation_id}")
        page = citation.get("page")

        if page is not None:
            label = f"[{display_name} p.{page}]"
        else:
            label = f"[{display_name}]"

        pdf_link, image_link = get_citation_links(citation)
        target_link = image_link or pdf_link

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
    label = get_citation_label(citation)
    pdf_link, image_link = get_citation_links(citation)
    reason = citation.get("reason") or ""
    excerpt = citation.get("excerpt") or ""
    used_in_answer = citation.get("used_in_answer", True)

    safe_label = html.escape(label)
    safe_reason = html.escape(reason)
    safe_excerpt = html.escape(excerpt)

    image_html = ""
    link_html = safe_label

    if image_link:
        safe_image_link = html.escape(image_link, quote=True)
        image_html = (
            f"<a href='{safe_image_link}' target='_blank'>"
            f"<img src='{safe_image_link}' "
            f"style='width:110px;"
            f"border:1px solid #ccc;"
            f"border-radius:4px;'>"
            f"</a>"
        )
        link_html = (
            f"<a href='{safe_image_link}' "
            f"target='_blank' "
            f"style='text-decoration:none;"
            f"font-weight:600;'>"
            f"{safe_label}</a>"
        )

    pdf_html = ""
    if pdf_link:
        pdf_html = (
            f"<a href='{html.escape(pdf_link, quote=True)}' "
            f"target='_blank'>PDFで開く</a>"
        )

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

    card = f"""
<div style="display:flex;gap:12px;align-items:flex-start;margin:10px 0 16px 0;">
  <div style="flex:0 0 auto;">{image_html}</div>
  <div style="flex:1;min-width:0;">
    <div>{link_html}{badge_html}</div>
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
book_display_name_map.clear()

for category, category_info in book_categories.items():
    for book_entry in category_info.get("books", []):
        full_name = book_entry["name"]
        display_name = book_entry.get("display_name", full_name)

        # UIの書籍選択用: display_name / full_name -> full_name
        book_name_map[display_name] = full_name
        book_name_map[full_name] = full_name

        # 回答本文のCitation表示用: full_name -> display_name
        book_display_name_map[full_name] = display_name

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
        st.session_state.question_submitted = True
        st.session_state.current_question = question
        st.session_state.mode = selected_mode


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
5. 同じチャットで質問を続けると、過去の会話と参照資料を踏まえて回答します。
6. 左側のチャット履歴から以前の調査を開き直して、その続きを質問できます。
7. 別テーマとして分けたい場合は「＋ 新しいチャット」を使用してください。
8. 出典には、書籍ページ・選定理由・該当箇所の抜粋・画像/PDFリンクを表示します。

## 各モード

### ルールブックに基づく回答と出典
- Vector/全文検索に加え、目次・索引を利用して関連ページを検索します。
- 「○頁参照」「次頁」など本文中の参照先も追跡します。
- 「表」「テーブル」「一覧」などの質問では、複数書籍にある表本体を追加探索します。
- 「槍→スピア」「流派→秘伝」のようなルール用語の検索展開も行います。
- 出典欄には、AIが回答中で実際に引用したページに加え、表本体や参照先など調査価値の高い関連資料を少数表示します。
- 同一チャットでは、会話履歴と過去に参照した資料を保持しながら追加質問できます。

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
# Search execution
# ============================================================

if st.session_state.get("question_submitted"):
    current_question = st.session_state.get("current_question", "")

    if not current_question:
        st.warning("質問を入力してください。")
    else:
        with st.spinner("AIが調査中です..."):
            try:
                current_mode = st.session_state.get("mode", "rules_strict")
                response = requests.post(
                    API_URL,
                    headers=get_api_headers(),
                    json={
                        "question": current_question,
                        "books": selected_books,
                        "model": selected_model,
                        "mode": current_mode,
                        "k": DEFAULT_HYBRID_K,
                        "chat_id": st.session_state.current_chat_id,
                    },
                    timeout=180,
                )

                if response.status_code == 200:
                    # 回答と出典はAPI側でチャットDBへ保存済み。
                    st.session_state.question_submitted = False
                    st.rerun()
                else:
                    st.error(
                        _api_error_detail(
                            response,
                            f"APIからの応答に失敗しました。コード: {response.status_code}",
                        )
                    )
            except Exception as exc:
                st.error(f"エラーが発生しました: {exc}")

    st.session_state.question_submitted = False


# ============================================================
# Chat history display
# ============================================================

# API応答後やチャット切替後の最新状態を取得する。
current_chat, current_chat_error = get_chat_detail(st.session_state.current_chat_id)
if current_chat_error:
    st.error(current_chat_error)
    st.stop()

st.caption(f"チャット: {current_chat.get('title') or '新しいチャット'}")

messages = current_chat.get("messages", [])
for message in messages:
    role = message.get("role")
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message.get("content", ""))
        continue

    metadata = message.get("metadata") or {}
    with st.chat_message("assistant"):
        display_text = link_citation_markers(
            message.get("content", ""),
            metadata.get("citations", []),
        )

        if metadata.get("model_used"):
            display_text += f"\n\n🔧 使用モデル: `{metadata['model_used']}`"

        if metadata.get("token_usage"):
            tokens = metadata["token_usage"]
            display_text += (
                "\n\n🧮 トークン数: "
                f"入力 {tokens.get('prompt_tokens', 0)}, "
                f"出力 {tokens.get('completion_tokens', 0)}, "
                f"合計 {tokens.get('total_tokens', 0)}"
            )
            input_price, output_price, total_price = calculate_price(
                metadata.get("model_used"),
                tokens,
            )
            display_text += (
                "\n\n💰 推定料金: "
                f"入力: ¥{input_price:.2f} / "
                f"出力: ¥{output_price:.2f} / "
                f"合計: ¥{total_price:.2f}"
            )

        st.markdown(display_text, unsafe_allow_html=True)

        if metadata.get("model_used") != "AIは使用していません":
            context_k = int(metadata.get("k_used", 0) or 0)
            hybrid_k = int(metadata.get("hybrid_k_used", 0) or 0)
            nav_pages = int(metadata.get("navigation_pages_used", 0) or 0)
            structured_pages = int(metadata.get("structured_pages_used", 0) or 0)
            reference_pages = int(metadata.get("reference_pages_used", 0) or 0)
            st.markdown(
                f"📊 推論コンテキスト: **{context_k} chunks** / "
                f"通常検索: **k={hybrid_k}** / "
                f"navigation: **{nav_pages} pages** / "
                f"表・一覧: **{structured_pages} pages** / "
                f"参照先: **{reference_pages} pages**"
            )

        citations = metadata.get("citations", [])
        if citations:
            used_citations = [
                citation for citation in citations
                if citation.get("used_in_answer", True)
            ]
            related_citations = [
                citation for citation in citations
                if not citation.get("used_in_answer", True)
            ]

            if used_citations:
                st.markdown("**📖 回答で使用した出典:**")
                for citation in used_citations:
                    render_citation(citation)

            if related_citations:
                st.markdown("**🔎 あわせて確認したい関連資料:**")
                st.caption(
                    "回答本文では直接引用していませんが、確認価値が高い関連ページです。"
                )
                for citation in related_citations:
                    render_citation(citation)
        elif metadata.get("sources"):
            st.markdown("**📖 出典:**")
            for source in metadata["sources"]:
                st.markdown(f"- {source}")

