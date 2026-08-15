# sw2.5-rule-bot

ソード・ワールド2.5のルール参照Botです。

このリポジトリはアプリケーションコードとDocker定義のみを管理します。OCRデータ、FAISSインデックス、PDF、ページ画像、`book_categories.json` は既存ホスト上のデータをbind mountして利用します。

## 構成

- `api`: FastAPI / RAG API
- `ui`: Streamlit UI
- `discord`: Discord Bot
- `pdfjs`: nginx + Mozilla PDF.js

Python系3サービスは同じDockerイメージを使用します。

## 初回セットアップ

```bash
git clone <repository-url> sw2.5-rule-bot
cd sw2.5-rule-bot
cp .env.example .env
```

`.env` を編集し、APIキーと既存データの絶対パスを設定してください。

最低限必要なデータは以下です。

```text
HOST_OCR_DIR          OCR JSONディレクトリ
HOST_VECTOR_INDEX_DIR FAISSインデックスディレクトリ
HOST_METADATA_DIR     各種データを含むディレクトリ
HOST_PDF_DIR          PDFファイルのディレクトリ
HOST_IMAGE_DIR        ページ画像のディレクトリ
```

既定値は現行環境を想定していますが、実際の配置に合わせて変更してください。

## 起動

```bash
docker compose up -d --build
```

既定ポート:

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- PDF.js: `http://localhost:8080/pdfjs/web/viewer.html`

## 公開URL

リバースプロキシや既存ドメインを使用する場合は `.env` の次の値を変更します。

```dotenv
PDFJS_BASE_URL=https://example.com/pdfjs/web/viewer.html
IMAGE_BASE_URL=https://example.com/image
```

## 更新

```bash
git pull
docker compose up -d --build
```

## Gitで管理しないもの

`.env`、認証情報、OCRデータ、FAISSインデックス、PDF、ページ画像はGitHubへ登録しません。

## PDF.js

Dockerビルド時にMozilla PDF.jsの配布zipを取得し、nginxから `/pdfjs/` として配信します。バージョンは `.env` の `PDFJS_VERSION` で固定できます。
