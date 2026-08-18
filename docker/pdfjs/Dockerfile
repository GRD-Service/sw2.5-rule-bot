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
# Paths
# ============================================================

viewer_html_path = Path("/out/pdfjs/web/viewer.html")
viewer_mjs_path = Path("/out/pdfjs/web/viewer.mjs")


# ============================================================
# 1. Inject viewer_defaults.js before viewer.mjs
# ============================================================

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
# 2. Tune PDF.js network loading
#
# PDF.js default:
#   rangeChunkSize = 64 KiB
#
# SW2.5:
#   rangeChunkSize = 1 MiB
#   disableAutoFetch = true
#   disableStream = true
#   disableRange = false
#
# disableAutoFetch requires disableStream=true in PDF.js.
#
# We patch the initial PDF open() call rather than relying on
# viewer URL parameters.
# ============================================================

viewer_mjs = viewer_mjs_path.read_text(
    encoding="utf-8",
)

open_pattern = re.compile(
    r'this\.open\(\{\s*url:\s*file\s*\}\);'
)

open_matches = list(
    open_pattern.finditer(viewer_mjs)
)

if len(open_matches) != 1:
    raise RuntimeError(
        "Expected exactly one initial PDF open call "
        "in viewer.mjs, "
        f"found {len(open_matches)}"
    )

network_replacement = """this.open({
          url: file,

          // SW2.5 PDF loading optimization.
          //
          // PDF.js defaults to 64 KiB range requests.
          // Large SW2.5 rulebooks caused hundreds of small
          // HTTP Range requests, resulting in browser-side
          // queueing/stalling even when Cloudflare returned HIT.
          //
          // Start with 1 MiB as the evaluation value.
          rangeChunkSize: 1024 * 1024,

          // Do not automatically fetch the whole PDF after
          // the currently required ranges have been loaded.
          //
          // PDF.js requires streaming to be disabled for
          // disableAutoFetch to work correctly.
          disableAutoFetch: true,
          disableStream: true,

          // Keep HTTP Range Requests enabled.
          disableRange: false,
        });"""

viewer_mjs = open_pattern.sub(
    network_replacement,
    viewer_mjs,
    count=1,
)

viewer_mjs_path.write_text(
    viewer_mjs,
    encoding="utf-8",
)


# ============================================================
# 3. Build-time verification
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


verify_mjs = viewer_mjs_path.read_text(
    encoding="utf-8"
)

required_network_markers = (
    "rangeChunkSize: 1024 * 1024",
    "disableAutoFetch: true",
    "disableStream: true",
    "disableRange: false",
)

for marker in required_network_markers:
    count = verify_mjs.count(marker)

    if count != 1:
        raise RuntimeError(
            "PDF.js network patch verification failed: "
            f"{marker!r} found {count} times"
        )


print("Verified viewer_defaults.js injection:")
print('<script src="viewer_defaults.js"></script>')
print(viewer_tag)

print("Verified PDF.js network tuning:")
for marker in required_network_markers:
    print(marker)
PY

RUN test -s /out/pdfjs/web/viewer_defaults.js \
    && grep -q \
       'viewer_defaults.js' \
       /out/pdfjs/web/viewer.html \
    && grep -q \
       'rangeChunkSize: 1024 \* 1024' \
       /out/pdfjs/web/viewer.mjs \
    && grep -q \
       'disableAutoFetch: true' \
       /out/pdfjs/web/viewer.mjs \
    && grep -q \
       'disableStream: true' \
       /out/pdfjs/web/viewer.mjs \
    && grep -q \
       'disableRange: false' \
       /out/pdfjs/web/viewer.mjs


FROM nginx:1.28-alpine

COPY --from=downloader \
     /out/pdfjs \
     /usr/share/nginx/html/pdfjs

RUN sed -i \
    's/application\/javascript[[:space:]]*js;/application\/javascript js mjs;/' \
    /etc/nginx/mime.types

COPY docker/pdfjs/default.conf \
     /etc/nginx/conf.d/default.conf