FROM alpine:3.22 AS downloader

ARG PDFJS_VERSION=5.7.284

ARG PDFJS_DEFAULT_ZOOM=page-fit
ARG PDFJS_DEFAULT_SCROLL_MODE=0
ARG PDFJS_DEFAULT_SPREAD_MODE=2

RUN apk add --no-cache curl unzip python3 \
    && mkdir -p /out/pdfjs \
    && curl -fL \
       "https://github.com/mozilla/pdf.js/releases/download/v${PDFJS_VERSION}/pdfjs-${PDFJS_VERSION}-dist.zip" \
       -o /tmp/pdfjs.zip \
    && unzip /tmp/pdfjs.zip -d /out/pdfjs

# ----------------------------------------------------------------------
# PDF.js viewer defaults
#
# webviewerloaded is dispatched after the viewer modules are loaded but
# before PDFViewerApplication initialization begins.
#
# We set our application defaults there.
# PDF.js subsequently loads its persistent Preferences from localStorage,
# so settings explicitly changed by the user take precedence.
# ----------------------------------------------------------------------

RUN cat > /out/pdfjs/web/viewer_defaults.js <<EOF
document.addEventListener("webviewerloaded", () => {
  const options = window.PDFViewerApplicationOptions;

  if (!options) {
    console.error(
      "PDF.js: PDFViewerApplicationOptions is not available."
    );
    return;
  }

  options.set("defaultZoomValue", "${PDFJS_DEFAULT_ZOOM}");
  options.set("scrollModeOnLoad", ${PDFJS_DEFAULT_SCROLL_MODE});
  options.set("spreadModeOnLoad", ${PDFJS_DEFAULT_SPREAD_MODE});

  console.info(
    "PDF.js defaults:",
    "zoom=${PDFJS_DEFAULT_ZOOM}",
    "scrollMode=${PDFJS_DEFAULT_SCROLL_MODE}",
    "spreadMode=${PDFJS_DEFAULT_SPREAD_MODE}"
  );
});
EOF

# viewer_defaults.js must be loaded before viewer.mjs.
#
# Insert it immediately before the PDF.js module script.
RUN python3 - <<'PY'
from pathlib import Path

path = Path("/out/pdfjs/web/viewer.html")
text = path.read_text(encoding="utf-8-sig")

marker = '<script src="viewer.mjs" type="module"></script>'

if marker not in text:
    marker = '<script type="module" src="viewer.mjs"></script>'

if marker not in text:
    raise RuntimeError(
        "Could not find viewer.mjs script element in viewer.html"
    )

replacement = (
    '<script src="viewer_defaults.js"></script>\n'
    '    ' + marker
)

text = text.replace(marker, replacement, 1)

path.write_text(text, encoding="utf-8")

print("Injected viewer_defaults.js before viewer.mjs")
PY


FROM nginx:1.28-alpine

COPY --from=downloader /out/pdfjs /usr/share/nginx/html/pdfjs

# nginx Alpine image does not map .mjs to JavaScript by default.
RUN sed -i \
    's/application\/javascript[[:space:]]*js;/application\/javascript js mjs;/' \
    /etc/nginx/mime.types

COPY docker/pdfjs/default.conf /etc/nginx/conf.d/default.conf