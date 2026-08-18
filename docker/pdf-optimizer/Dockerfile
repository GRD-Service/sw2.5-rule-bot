FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       qpdf \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install \
       PyMuPDF==1.26.3 \
       Pillow==11.3.0 \
       pikepdf

COPY scripts/optimize_pdfs.py /app/scripts/optimize_pdfs.py
COPY scripts/rebalance_page_tree.py /app/scripts/rebalance_page_tree.py

CMD ["python", "/app/scripts/optimize_pdfs.py"]