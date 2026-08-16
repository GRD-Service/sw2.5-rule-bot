FROM alpine:3.22 AS downloader

ARG PDFJS_VERSION=5.7.284

ARG PDFJS_DEFAULT_ZOOM=page-fit
ARG PDFJS_DEFAULT_SCROLL_MODE=0
ARG PDFJS_DEFAULT_SPREAD_MODE=2

RUN apk add --no-cache curl unzip \
    && mkdir -p /out/pdfjs \
    && curl -fL \
       "https://github.com/mozilla/pdf.js/releases/download/v${PDFJS_VERSION}/pdfjs-${PDFJS_VERSION}-dist.zip" \
       -o /tmp/pdfjs.zip \
    && unzip /tmp/pdfjs.zip -d /out/pdfjs

RUN cat > /out/pdfjs/web/viewer_defaults.js <<EOF
(() => {
  const STORAGE_KEY = "pdfjs.preferences";

  const defaults = {
    defaultZoomValue: "${PDFJS_DEFAULT_ZOOM}",
    scrollModeOnLoad: ${PDFJS_DEFAULT_SCROLL_MODE},
    spreadModeOnLoad: ${PDFJS_DEFAULT_SPREAD_MODE},
  };

  let preferences = {};

  try {
    const stored = localStorage.getItem(STORAGE_KEY);

    if (stored) {
      preferences = JSON.parse(stored) || {};
    }
  } catch (error) {
    console.warn(
      "PDF.js: failed to read stored preferences:",
      error
    );
    return;
  }

  let changed = false;

  for (const [name, value] of Object.entries(defaults)) {
    if (!Object.prototype.hasOwnProperty.call(preferences, name)) {
      preferences[name] = value;
      changed = true;
    }
  }

  if (changed) {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(preferences)
      );

      console.info(
        "PDF.js initial preferences installed:",
        defaults
      );
    } catch (error) {
      console.warn(
        "PDF.js: failed to install initial preferences:",
        error
      );
    }
  } else {
    console.info(
      "PDF.js existing preferences preserved:",
      {
        defaultZoomValue: preferences.defaultZoomValue,
        scrollModeOnLoad: preferences.scrollModeOnLoad,
        spreadModeOnLoad: preferences.spreadModeOnLoad,
      }
    );
  }
})();
EOF

RUN sed -i \
    '0,/<script src="viewer.mjs" type="module"><\/script>/s//<script src="viewer_defaults.js"><\/script>\n    <script src="viewer.mjs" type="module"><\/script>/' \
    /out/pdfjs/web/viewer.html


FROM nginx:1.28-alpine

COPY --from=downloader /out/pdfjs /usr/share/nginx/html/pdfjs

RUN sed -i \
    's/application\/javascript[[:space:]]*js;/application\/javascript js mjs;/' \
    /etc/nginx/mime.types

COPY docker/pdfjs/default.conf /etc/nginx/conf.d/default.conf