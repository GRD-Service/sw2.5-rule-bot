FROM alpine:3.22 AS downloader

ARG PDFJS_VERSION=5.7.284

RUN apk add --no-cache curl unzip python3 \
    && mkdir -p /out/pdfjs \
    && curl -fL \
       "https://github.com/mozilla/pdf.js/releases/download/v${PDFJS_VERSION}/pdfjs-${PDFJS_VERSION}-dist.zip" \
       -o /tmp/pdfjs.zip \
    && unzip /tmp/pdfjs.zip -d /out/pdfjs

COPY docker/pdfjs/viewer_defaults.js \
     /out/pdfjs/web/viewer_defaults.js

RUN python3 - <<'PY'
from pathlib import Path
import re


# ============================================================
# Inject viewer_defaults.js before viewer.mjs
# ============================================================

viewer_html_path = Path("/out/pdfjs/web/viewer.html")

text = viewer_html_path.read_text(encoding="utf-8-sig")

# viewer.mjs の script タグを属性順に依存せず探す。
pattern = re.compile(
    r'<script\b'
    r'(?=[^>]*\bsrc=["\']viewer\.mjs["\'])'
    r'(?=[^>]*\btype=["\']module["\'])'
    r'[^>]*></script>',
    re.IGNORECASE,
)

matches = list(pattern.finditer(text))

if len(matches) != 1:
    raise RuntimeError(
        "Expected exactly one viewer.mjs script tag, "
        f"found {len(matches)}"
    )

viewer_tag = matches[0].group(0)

replacement = (
    '<script src="viewer_defaults.js"></script>\n'
    '    ' + viewer_tag
)

text = (
    text[:matches[0].start()]
    + replacement
    + text[matches[0].end():]
)

viewer_html_path.write_text(
    text,
    encoding="utf-8",
)


# ============================================================
# Build-time verification
# ============================================================

verify_html = viewer_html_path.read_text(
    encoding="utf-8"
)

if (
    '<script src="viewer_defaults.js"></script>'
    not in verify_html
):
    raise RuntimeError(
        "viewer_defaults.js injection failed"
    )

if (
    verify_html.index("viewer_defaults.js")
    > verify_html.index("viewer.mjs")
):
    raise RuntimeError(
        "viewer_defaults.js must be loaded "
        "before viewer.mjs"
    )

print("Verified viewer_defaults.js injection:")
print('<script src="viewer_defaults.js"></script>')
print(viewer_tag)
PY

RUN test -s /out/pdfjs/web/viewer_defaults.js \
    && grep -q \
       'viewer_defaults.js' \
       /out/pdfjs/web/viewer.html


FROM nginx:1.28-alpine

COPY --from=downloader \
     /out/pdfjs \
     /usr/share/nginx/html/pdfjs

RUN sed -i \
    's/application\/javascript[[:space:]]*js;/application\/javascript js mjs;/' \
    /etc/nginx/mime.types

COPY docker/pdfjs/default.conf \
     /etc/nginx/conf.d/default.conf