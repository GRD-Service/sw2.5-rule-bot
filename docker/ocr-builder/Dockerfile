FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install \
       google-cloud-vision==3.10.2

COPY scripts/google_vision_ocr.py \
     /app/scripts/google_vision_ocr.py

CMD [
    "python",
    "/app/scripts/google_vision_ocr.py"
]