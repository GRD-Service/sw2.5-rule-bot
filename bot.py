import discord
from discord.ext import commands
from discord import app_commands
import requests
import os
from dotenv import load_dotenv
import json
from get_page_link import get_page_link, auto_link_answer_text

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("QA_API_URL", "http://localhost:8000/ask")

# --- カテゴリ別の書籍を読み込み ---
def load_book_categories():
    try:
        with open(os.getenv("BOOK_CATEGORY_PATH", "./book/book_categories.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"カテゴリ情報の読み込みに失敗しました: {e}")
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

class TRPGBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        # スラッシュコマンドの同期を確認する
        print("✅ スラッシュコマンド同期を準備中...")
        for guild in self.guilds:
            await self.tree.sync(guild=guild)  # サーバーごとに同期を行う
            print(f"✅ コマンド同期完了: {guild.name}")
        print("✅ コマンド同期完了")

bot = TRPGBot()

@bot.event
async def on_ready():
    print(f"✅ Bot準備完了: {bot.user}")
    await bot.setup_hook()  # 必要なタイミングで同期を実行

# AI Botモード (ask)
@bot.tree.command(name="ask", description="TRPGルールをAI Botモードで質問する")
@app_commands.describe(question="質問内容を入力してください")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()

    # AI Botモード
    payload = {
        "question": question,
        "books": [],
        "mode": "rules_strict"
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        answer = result.get("answer", "回答が見つかりませんでした。")
        sources = result.get("sources", [])
        model_used = result.get("model_used", "不明")
        token_usage = result.get("token_usage", {})

        enriched_answer = auto_link_answer_text(answer, book_name_map)

        links = []
        for src in sources:
            if " - p." in src:
                book, page_str = src.split(" - p.")
                book = book.split(" / ")[-1].strip()
                try:
                    page = int(page_str)
                    pdf_link, image_link = get_page_link(book, page)
                    links.append(f"[{book} - p.{page}]({image_link}) [📄 PDFで開く]({pdf_link})")
                except:
                    links.append(src)
            else:
                links.append(src)

        reply_parts = []
        reply_parts.append(f"**❓ 質問:** {question}")

        max_length = 1800
        answer_lines = enriched_answer.splitlines()
        current_chunk = ""
        for line in answer_lines:
            if len(current_chunk) + len(line) + 1 > max_length:
                reply_parts.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk.strip():
            reply_parts.append(current_chunk.strip())

        if links:
            chunk = "**📖 出典:**\n"
            for link in links:
                if len(chunk) + len(link) + 2 > max_length:
                    reply_parts.append(chunk.strip())
                    chunk = "**📖 出典:**\n" + link + "\n"
                else:
                    chunk += link + "\n"
            if chunk.strip():
                reply_parts.append(chunk.strip())

        # 使用モデルとトークン数も追加
        token_info = ""
        if token_usage:
            token_info = f"🧮 トークン数: 入力 {token_usage.get('prompt_tokens', 0)}, 出力 {token_usage.get('completion_tokens', 0)}, 合計 {token_usage.get('total_tokens', 0)}"

        reply_parts.append(f"🔧 使用モデル: `{model_used}`")
        if token_info:
            reply_parts.append(token_info)

        for part in reply_parts:
            if part.strip():
                await interaction.followup.send(part.strip()[:2000])

    except Exception as e:
        await interaction.followup.send(f"❌ API呼び出し中にエラーが発生しました: `{e}`")

# 全文検索モード (search)
@bot.tree.command(name="search", description="TRPGルールを全文検索モードで質問する")
@app_commands.describe(question="質問内容を入力してください")
async def search(interaction: discord.Interaction, question: str):
    await interaction.response.defer()

    # 全文検索モード
    payload = {
        "question": question,
        "books": [],
        "mode": "exact_search"
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        answer = result.get("answer", "全文検索結果が見つかりませんでした。")
        sources = result.get("sources", [])
        model_used = result.get("model_used", "AIは使用していません")
        token_usage = result.get("token_usage", {})

        links = []
        for src in sources:
            if " - p." in src:
                book, page_str = src.split(" - p.")
                book = book.split(" / ")[-1].strip()
                try:
                    page = int(page_str)
                    pdf_link, image_link = get_page_link(book, page)
                    links.append(f"[{book} - p.{page}]({image_link}) [📄 PDFで開く]({pdf_link})")
                except:
                    links.append(src)
            else:
                links.append(src)

        reply_parts = []
        reply_parts.append(f"**🔍 検索結果:** {answer}")

        max_length = 1800
        if links:
            chunk = "**📖 出典:**\n"
            for link in links:
                if len(chunk) + len(link) + 2 > max_length:
                    reply_parts.append(chunk.strip())
                    chunk = "**📖 出典:**\n" + link + "\n"
                else:
                    chunk += link + "\n"
            if chunk.strip():
                reply_parts.append(chunk.strip())

        # 使用モデルとトークン数も追加
        token_info = ""
        if token_usage:
            token_info = f"🧮 トークン数: 入力 {token_usage.get('prompt_tokens', 0)}, 出力 {token_usage.get('completion_tokens', 0)}, 合計 {token_usage.get('total_tokens', 0)}"

        reply_parts.append(f"🔧 使用モデル: `{model_used}`")
        if token_info:
            reply_parts.append(token_info)

        for part in reply_parts:
            if part.strip():
                await interaction.followup.send(part.strip()[:2000])

    except Exception as e:
        await interaction.followup.send(f"❌ API呼び出し中にエラーが発生しました: `{e}`")

bot.run(DISCORD_TOKEN)
