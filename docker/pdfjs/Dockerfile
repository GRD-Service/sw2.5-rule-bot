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

COPY docker/pdfjs/patch_defaults.py /tmp/patch_defaults.py

RUN python3 /tmp/patch_defaults.py \
      /out/pdfjs/web/viewer.mjs \
      "${PDFJS_DEFAULT_ZOOM}" \
      "${PDFJS_DEFAULT_SCROLL_MODE}" \
      "${PDFJS_DEFAULT_SPREAD_MODE}"


FROM nginx:1.28-alpine

COPY --from=downloader /out/pdfjs /usr/share/nginx/html/pdfjs

# nginx Alpine image does not map .mjs to JavaScript by default.
RUN sed -i \
    's/application\/javascript[[:space:]]*js;/application\/javascript js mjs;/' \
    /etc/nginx/mime.types

COPY docker/pdfjs/default.conf /etc/nginx/conf.d/default.conf