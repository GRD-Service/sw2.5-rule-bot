import streamlit as st
import streamlit.components.v1 as components
import requests
import os
import json
import re
from get_page_link import (
    get_citation_label,
    get_citation_links,
)

def link_citation_markers(
    answer: str,
    citations: list,
) -> str:
    """
    回答中の [C1] 形式の引用IDを、
    構造化citationに基づくリンクへ変換する。
    """

    citation_map = {
        citation.get("id"): citation
        for citation in citations
        if citation.get("id") is not None
    }

    def replacer(match):
        citation_id = int(
            match.group(1)
        )

        citation = citation_map.get(
            citation_id
        )

        if not citation:
            return match.group(0)

        label = get_citation_label(
            citation
        )

        pdf_link, image_link = (
            get_citation_links(
                citation
            )
        )

        if image_link and pdf_link:
            return (
                f"（<a href='{image_link}' "
                f"target='_blank'>{label}</a> "
                f"[<a href='{pdf_link}' "
                f"target='_blank'>PDF</a>]）"
            )

        return f"（{label}）"

    return re.sub(
        r"\[C(\d+)\]",
        replacer,
        answer,
    )

st.set_page_config(page_title="ソード・ワールド2.5 ルールAI bot", layout="wide")
st.title("📚 ソード・ワールド2.5 ルールAI bot")

API_URL = os.getenv("QA_API_URL", "http://localhost:8000/ask")

if "history" not in st.session_state:
    st.session_state.history = []
if "question_submitted" not in st.session_state:
    st.session_state.question_submitted = False

# --- カテゴリ別の書籍を読み込み ---
def load_book_categories():
    try:
        with open(os.getenv("BOOK_CATEGORY_PATH", "./book/book_categories.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.sidebar.error(f"カテゴリ情報の読み込みに失敗しました: {e}")
        return {}

book_categories = load_book_categories()

# --- 書籍名を平坦化して出典参照に備える ---
book_name_map = {}
for category, category_info in book_categories.items():
    for book_entry in category_info["books"]:
        full_name = book_entry["name"]
        display_name = book_entry.get("display_name", full_name)
        book_name_map[display_name] = full_name
        book_name_map[full_name] = full_name

# --- サイドバー：検索条件と書籍カテゴリ ---
st.sidebar.header("🔍 検索条件")

# --- モデルごとの価格設定（1kトークンあたり） ---
model_prices = {
    "gpt-5.4-nano": {
        "input": 0.20,
        "output": 1.25,
    },
    "gpt-5.4-mini": {
        "input": 0.75,
        "output": 4.50,
    },
    "gpt-5.4": {
        "input": 2.50,
        "output": 15.00,
    },
}

# --- トークン価格計算 ---
def calculate_price(model, token_usage):
    price_info = model_prices.get(model, {"input": 0, "output": 0})
    ratio = 150
    input_price = (token_usage.get("prompt_tokens", 0) / 1000000) * price_info["input"] * ratio
    output_price = (token_usage.get("completion_tokens", 0) / 1000000) * price_info["output"] * ratio
    total_price = input_price + output_price
    return input_price, output_price, total_price

# モデル選択肢の内部値と表示用のラベルを対応させた辞書
model_options = {
    "gpt-5.4-nano": "GPT-5.4 Nano (通常)",
    "gpt-5.4-mini": "GPT-5.4 Mini (高性能)",
    "gpt-5.4": "GPT-5.4 (最高性能)",
}

# 表示用のリストを作成（内部値を表示用にマッピング）
display_model_options = list(model_options.values())

# デフォルトで選択するモデルを設定
default_model_display = model_options["gpt-5.4-nano"]

# モデル選択
selected_model_display = st.sidebar.selectbox("🧠 モデル選択", display_model_options, index=display_model_options.index(default_model_display))
selected_model = next(key for key, value in model_options.items() if value == selected_model_display)

st.sidebar.markdown("📚 **検索対象の書籍**")

selected_books = []
for category, category_info in book_categories.items():
    books = category_info["books"]
    default_checked = category_info.get("default_enabled", True)
    with st.sidebar.expander(category, expanded=True):
        toggle_key = f"toggle_{category}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = default_checked

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

# 操作説明のセクション
with st.expander("操作説明 (クリックして開く)"):
    st.markdown("""
    ## このアプリケーションの使い方

    1. **質問入力欄**に質問を入力してください。  
    2. 回答モードを選択してください。
    3. 必要に応じて選択条件を選択してください。
    4. 質問をするをクリックすると、回答が表示されます。
    5. 出典がある場合や、全文検索では、該当ページがサムネイルと共に提示されます。
    
    ## 各モードの使い方
    
    ### ルールブックに基づく回答と出典
    
    - 質問入力欄に入力された事項について、どのルールブックのどのページが該当するかを推測し、AIにデータを渡すことで推論を行い、回答と出典を提示します。
    - 検索条件の、モデル選択と検索対象の書籍の選択が有効です。
    - 表でまとめられた内容を検索することが苦手です。例えば`コンジャラー技能をレベル5から6に上げるのに必要な経験点は？`という質問は回答できません。`経験値テーブルはどこに記載されてますか？`と質問し、自身で表を確認してください。
    - 「コンテキストには存在しない」と回答があった場合、どのルールブックのどのページが該当するかの推測に失敗している可能性があります。もう少し具体的に質問をするか、「検索範囲を広げて再質問」をクリックすると、事前の推測の範囲を拡大して再度AIに回答させます。ただし、範囲を拡大すると、AIの利用料金が急激に増加する可能性がありますので、ご注意下さい。
    - 正式名称から外れると、急に精度が下がります。`運命変転`と聞いても的外れな回答しかありませんが、`剣の加護/運命変転`と聞くと適切な回答が得られます。なお、正式名称が分からない場合は、具体的に何の話なのかを付加すると回答の確率が上がります。`人間の種族特性としての運命変転`と聞くと適切な回答が得られます。
    
    ### 全文検索モード
    
    - 質問入力欄に入力されたキーワードについて、どのルールブックのどのページにその文字が出現するかを回答します。
    - AIは使用していません。純粋な全文検索となります。
    - 検索条件の、モデルの選択は無効です。検索対象の書籍の選択は有効です。
    - スペースで区切ると、絞込検索(AND検索)が可能です。
    
    ### AI自由解釈モード
    
    - 質問入力欄に入力された事項について、AIに推論を行わせます。
    - AIには、推論はソードワールド2.5に関する事項に限定するように指示を出していますで、ユーザーが意識する必要は有りません。ソードワールド2.5の事で聞きたいことをそのまま入力してください。
    - 本システムが保有しているルールブックのデータは、AIには渡していませんので、AIの知りうる範囲での回答となります。
    - 出典・掲載ページに関する質問は、ほとんど回答できません。
    - 検索条件の、検索対象の書籍の選択は無効です。モデル選択は有効です。

    ## 注意点
    - モデル選択は基本的には`GPT-5.4 Nano`を使用して下さい。
    - うまく回答が出ない場合は`GPT-5.4 Mini`、
    - さらに高い精度が必要な場合は`GPT-5.4`を使用してください。
    - サムネイルや書籍名をクリックすると、該当ページの画像が見えます。PDFのリンクもありますが、書籍がかなり重いので、快適に使うのは難しいかも知れません。
    
    """)

# --- 質問入力フォーム ---
with st.form("question_form"):
    question = st.text_input("質問", placeholder="例: マルチアクションはどういった戦闘特技ですか？", label_visibility="collapsed")

    mode_display = st.radio(
        "回答モード選択",
        ["🛡️ ルールブックに基づく回答と出典", "🔍 全文検索モード", "💬 AI自由解釈モード"],
        index=0
    )

    mode_map = {
        "🛡️ ルールブックに基づく回答と出典": "rules_strict",
        "🔍 全文検索モード": "exact_search",
        "💬 AI自由解釈モード": "free_chat",
    }
    selected_mode = mode_map[mode_display]

    submitted = st.form_submit_button("💬 質問する")
    if submitted:
        st.session_state.k_used = 10
        st.session_state.question_submitted = True
        st.session_state.current_question = question
        st.session_state.mode = selected_mode
        
# --- 検索実行 ---
if st.session_state.get("question_submitted"):
    question = st.session_state.get("current_question", "")
    if not question:
        st.warning("質問を入力してください。")
    else:
        with st.spinner("AIが調査中です..."):
            try:
                response = requests.post(API_URL, json={
                    "question": question,
                    "books": selected_books,
                    "model": selected_model,
                    "mode": st.session_state.get("mode", "rules_strict")
                })
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.history.append({
                        "question": question,
                        "answer": result["answer"],

                        # 新API
                        "citations": result.get(
                            "citations",
                            [],
                        ),

                        # 旧API互換
                        "sources": result.get(
                            "sources",
                            [],
                        ),

                        "model_used": result.get(
                            "model_used"
                        ),
                        "token_usage": result.get(
                            "token_usage",
                            {},
                        ),
                        "k_used": result.get(
                            "k_used",
                            10,
                        ),
                        "max_k": result.get(
                            "max_k",
                            50,
                        ),
                    })

                    # 検索結果が少ない場合、再検索ボタンを表示
                    k_used = result.get("k_used", 10)
                    max_k = result.get("max_k", 50)

                    # k_used < max_k なら再検索ボタンを表示
                    if k_used < max_k:
                        st.session_state.search_button_visible = True  # 再検索ボタンを表示するフラグ
                    else:
                        st.session_state.search_button_visible = False  # 再検索ボタンを非表示

                else:
                    st.error("APIからの応答に失敗しました。コード: {}".format(response.status_code))

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
        st.session_state.question_submitted = False

# --- 再検索処理 --- 
def expand_search():
    # 現在のkの値を取得
    if 'k_used' not in st.session_state:
        st.session_state.k_used = 10  # 初回はk=10に設定

    # kを10増加
    new_k = st.session_state.k_used + 10
    max_k = 50  # 最大k

    if new_k <= max_k:
        # session_state内のk_usedを更新
        st.session_state.k_used = new_k  # 次回の検索のためにkを更新

        # 再検索リクエストを送信
        response = requests.post(API_URL, json={
            "question": st.session_state.get("current_question", ""),
            "books": selected_books,
            "model": selected_model,
            "mode": st.session_state.get("mode", "rules_strict"),
            "k": new_k  # 新しいkを使って再検索
        })
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.history.append({
                "question": st.session_state.get("current_question", ""),
                "answer": result["answer"],
                "citations": result.get("citations", []),
                "sources": result.get("sources", []),
                "model_used": result.get("model_used"),
                "token_usage": result.get("token_usage", {}),
                "k_used": st.session_state.k_used,  # 使用されたk
                "max_k": max_k  # max_kも取得して表示
            })
        else:
            st.error("再質問に失敗しました。")
    else:
        st.warning(f"最大 k 値を超えました: {max_k}")

# --- チャット風履歴表示 --- 
for idx, entry in enumerate(reversed(st.session_state.history)):
    with st.chat_message("Q"):
        st.markdown(entry["question"])
    with st.chat_message("A"):
        display_text = link_citation_markers(entry["answer"], entry.get("citations", []))
        if entry.get("model_used"):
            display_text += f"\n\n🔧 使用モデル: `{entry['model_used']}`"
        if entry.get("token_usage"):
            tokens = entry["token_usage"]
            display_text += f"\n\n🧮 トークン数: 入力 {tokens.get('prompt_tokens', 0)}, 出力 {tokens.get('completion_tokens', 0)}, 合計 {tokens.get('total_tokens', 0)}"
            input_price, output_price, total_price = calculate_price(entry["model_used"], tokens)
            display_text += f"\n\n💰 推定料金: 入力: ¥{input_price:.2f} / 出力: ¥{output_price:.2f} / 合計: ¥{total_price:.2f}"

        st.markdown(display_text, unsafe_allow_html=True)

        # 使用されたkとmax_kを表示
        if entry.get("k_used"):
            k_used = entry.get("k_used", 10)
            max_k = entry.get("max_k", 50)
            display_text = f"\n\n📊 事前検索範囲 {k_used} / {max_k}"
            st.markdown(display_text, unsafe_allow_html=True)

            # 再検索ボタンを「回答」部分と「使用モデル」の間に表示
            if k_used < max_k:
                # `st.session_state.search_button_visible` の状態を管理
                if "search_button_visible" not in st.session_state:
                    st.session_state.search_button_visible = True

                if st.session_state.search_button_visible:
                    st.button("事前検索範囲を広げて再質問", key=f"expand_search_{idx}", on_click=expand_search)

        citations = entry.get(
            "citations",
            [],
        )

        if citations:
            st.markdown("**📖 出典:**")

            for citation in citations:
                try:
                    label = get_citation_label(
                        citation
                    )

                    pdf_link, image_link = (
                        get_citation_links(
                            citation
                        )
                    )

                    if pdf_link and image_link:
                        thumbnail_html = f"""
        <div style="margin-bottom:10px;">
        <a href="{image_link}"
            target="_blank"
            style="text-decoration:none;">
            <img src="{image_link}"
                style="width:100px;border:1px solid #ccc;">
            <div style="
                display:inline-block;
                margin-left:10px;
                vertical-align:middle;">
            {label}
            </div>
        </a>
        [<a href="{pdf_link}"
            target="_blank">PDF</a>]
        </div>
        """

                        st.markdown(
                            thumbnail_html,
                            unsafe_allow_html=True,
                        )

                    else:
                        st.markdown(
                            f"- {label}"
                        )

                except Exception:
                    st.markdown(
                        f"- {citation}"
                    )

        # 旧API互換
        elif entry.get("sources"):
            st.markdown("**📖 出典:**")

            for src in entry["sources"]:
                st.markdown(
                    f"- {src}"
                )

        st.markdown("---")

# --- サイドバー履歴クリア ---
st.sidebar.markdown("---")
if st.sidebar.button("🧹 履歴をクリア"):
    st.session_state.history.clear()
    st.experimental_rerun()
