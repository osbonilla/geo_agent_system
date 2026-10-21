# ══════════════════════════════════════════════════════════
#  Backend — FastAPI + LangGraph + ChromaDB client
# ══════════════════════════════════════════════════════════
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 lo pide onnxruntime (embeddings locales de Chroma);
# libpq5 lo pide psycopg2; el resto son dependencias de arcgis.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq5 \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Capa de dependencias aparte: cambiar el código no reinstala todo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y

COPY . .

# Usuario sin privilegios
RUN useradd --create-home --uid 1000 geo \
    && mkdir -p /app/chroma_data \
    && chown -R geo:geo /app
USER geo

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
