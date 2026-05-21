FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data/raw ./data/raw
COPY data/index/.gitkeep ./data/index/.gitkeep
COPY scripts ./scripts
COPY start.sh .

RUN python scripts/ingest_documents.py

EXPOSE 8000

CMD ["bash", "start.sh"]
