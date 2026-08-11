FROM alpine:3.22 AS downloader

ARG PDFJS_VERSION=6.2.108
RUN apk add --no-cache curl unzip \
    && mkdir -p /out/pdfjs \
    && curl -fL "https://github.com/mozilla/pdf.js/releases/download/v${PDFJS_VERSION}/pdfjs-${PDFJS_VERSION}-dist.zip" -o /tmp/pdfjs.zip \
    && unzip /tmp/pdfjs.zip -d /out/pdfjs

FROM nginx:1.28-alpine
COPY --from=downloader /out/pdfjs /usr/share/nginx/html/pdfjs
COPY docker/pdfjs/default.conf /etc/nginx/conf.d/default.conf
