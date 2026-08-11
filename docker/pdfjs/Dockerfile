FROM alpine:3.22 AS downloader

ARG PDFJS_VERSION=5.1.91

RUN apk add --no-cache curl unzip \
    && mkdir -p /out/pdfjs \
    && curl -fL \
       "https://github.com/mozilla/pdf.js/releases/download/v${PDFJS_VERSION}/pdfjs-${PDFJS_VERSION}-dist.zip" \
       -o /tmp/pdfjs.zip \
    && unzip /tmp/pdfjs.zip -d /out/pdfjs


FROM nginx:1.28-alpine

COPY --from=downloader /out/pdfjs /usr/share/nginx/html/pdfjs

# nginx Alpine image does not map .mjs to JavaScript by default.
RUN sed -i \
    's/application\/javascript[[:space:]]*js;/application\/javascript js mjs;/' \
    /etc/nginx/mime.types

COPY docker/pdfjs/default.conf /etc/nginx/conf.d/default.conf