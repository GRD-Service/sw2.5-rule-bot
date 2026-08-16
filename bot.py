import json
import os
import re

import discord
import requests
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from get_page_link import (
    get_citation_label,
    get_citation_links,
)


load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("QA_API_URL", "http://localhost:8000/ask")


# ============================================================
# Book categories
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
        print(f"カテゴリ情報の読み込みに失敗しました: {exc}")
        return {}


book_categories = load_book_categories()
book_name_map = {}
for category, category_info in book_categories.items():
    for book_entry in category_info.get("books", []):
        full_name = book_entry["name"]
        display_name = book_entry.get("display_name", full_name)
        book_name_map[display_name] = full_name
        book_name_map[full_name] = full_name


# ============================================================
# Citation helpers
# ============================================================


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
            return f"[{label}]({image_link}) [[PDF]]({pdf_link})"
        return label

    return re.sub(r"\[C(\d+)\]", replacer, answer)


def citation_to_discord_text(citation: dict) -> str:
    label = get_citation_label(citation)
    pdf_link, image_link = get_citation_links(citation)
    reason = (citation.get("reason") or "").strip()
    excerpt = (citation.get("excerpt") or "").strip()

    if image_link:
        title = f"[{label}]({image_link})"
    else:
        title = label

    if pdf_link:
        title += f" [📄 PDF]({pdf_link})"

    parts = [f"**{title}**"]
    if reason:
        parts.append(f"選定理由: {reason}")
    if excerpt:
        parts.append(f"> {excerpt}")
    return "\n".join(parts)


def chunk_messages(lines: list[str], max_length: int = 1800) -> list[str]:
    result = []
    current = ""

    for line in lines:
        addition = line + "\n"
        if len(current) + len(addition) > max_length:
            if current.strip():
                result.append(current.strip())
            current = addition
        else:
            current += addition

    if current.strip():
        result.append(current.strip())
    return result


# ============================================================
# Discord bot
# ============================================================


class TRPGBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        print("✅ スラッシュコマンド同期を準備中...")
        await self.tree.sync()
        print("✅ スラッシュコマンド同期完了")


bot = TRPGBot()


@bot.event
async def on_ready():
    print(f"✅ Bot準備完了: {bot.user}")


# ============================================================
# /ask
# ============================================================


@bot.tree.command(name="ask", description="TRPGルールをAI Botモードで質問する")
@app_commands.describe(question="質問内容を入力してください")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()

    payload = {
        "question": question,
        "books": [],
        "mode": "rules_strict",
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()

        answer = result.get("answer", "回答が見つかりませんでした。")
        citations = result.get("citations", [])
        sources = result.get("sources", [])
        model_used = result.get("model_used", "不明")
        token_usage = result.get("token_usage", {})
        context_k = result.get("k_used", 0)
        hybrid_k = result.get("hybrid_k_used", 0)
        nav_pages = result.get("navigation_pages_used", 0)

        enriched_answer = link_citation_markers(answer, citations)

        reply_parts = [f"**❓ 質問:** {question}"]
        reply_parts.extend(chunk_messages(enriched_answer.splitlines()))

        if citations:
            source_blocks = [citation_to_discord_text(citation) for citation in citations]
            source_lines = ["**📖 回答で使用した出典:**"]
            for block in source_blocks:
                source_lines.append(block)
                source_lines.append("")
            reply_parts.extend(chunk_messages(source_lines))
        elif sources:
            source_lines = ["**📖 出典:**"] + [f"- {src}" for src in sources]
            reply_parts.extend(chunk_messages(source_lines))

        reply_parts.append(f"🔧 使用モデル: `{model_used}`")

        if context_k or hybrid_k or nav_pages:
            reply_parts.append(
                f"📊 推論コンテキスト: {context_k} chunks / "
                f"通常検索: k={hybrid_k} / navigation補完: {nav_pages} pages"
            )

        if token_usage:
            reply_parts.append(
                "🧮 トークン数: "
                f"入力 {token_usage.get('prompt_tokens', 0)}, "
                f"出力 {token_usage.get('completion_tokens', 0)}, "
                f"合計 {token_usage.get('total_tokens', 0)}"
            )

        for part in reply_parts:
            if part.strip():
                await interaction.followup.send(part.strip()[:2000])

    except Exception as exc:
        await interaction.followup.send(
            f"❌ API呼び出し中にエラーが発生しました: `{exc}`"
        )


# ============================================================
# /search
# ============================================================


@bot.tree.command(name="search", description="TRPGルールを全文検索モードで質問する")
@app_commands.describe(question="質問内容を入力してください")
async def search(interaction: discord.Interaction, question: str):
    await interaction.response.defer()

    payload = {
        "question": question,
        "books": [],
        "mode": "exact_search",
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()

        answer = result.get("answer", "全文検索結果が見つかりませんでした。")
        citations = result.get("citations", [])
        sources = result.get("sources", [])
        model_used = result.get("model_used", "AIは使用していません")

        reply_parts = [f"**🔍 検索結果:** {answer}"]

        if citations:
            source_lines = ["**📖 出典:**"]
            for citation in citations:
                source_lines.append(citation_to_discord_text(citation))
                source_lines.append("")
            reply_parts.extend(chunk_messages(source_lines))
        elif sources:
            reply_parts.extend(
                chunk_messages(["**📖 出典:**"] + [f"- {src}" for src in sources])
            )

        reply_parts.append(f"🔧 使用モデル: `{model_used}`")

        for part in reply_parts:
            if part.strip():
                await interaction.followup.send(part.strip()[:2000])

    except Exception as exc:
        await interaction.followup.send(
            f"❌ API呼び出し中にエラーが発生しました: `{exc}`"
        )


bot.run(DISCORD_TOKEN)
