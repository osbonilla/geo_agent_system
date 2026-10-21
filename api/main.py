"""
api/main.py
────────────
Servidor FastAPI del sistema multi-agente geoespacial.

  POST /api/chat            → preguntar a los agentes
  GET  /api/webmaps         → WebMaps de ArcGIS Online
  POST /api/ingest-webmap   → vectorizar un WebMap
  GET  /api/status          → estado de LLM, ArcGIS, PostgreSQL y ChromaDB
  GET  /api/llm-providers   → proveedores de LLM soportados
  POST /api/llm-select      → cambiar de modelo en caliente
  GET  /health              → healthcheck para Docker

Levantar:
  uvicorn api.main:app --reload --port 8000
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes.chat import router as chat_router
from api.routes.status import router as status_router
from api.routes.webmaps import router as webmaps_router
from tools.utils import setup_logging
from config.settings import get_settings

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Arranque tolerante: si algo falta, la API igual levanta y lo reporta
    en /api/status. Nada de morir en el import por una key vacía.
    """
    logger.info("Iniciando sistema multi-agente geoespacial...")

    from db.models import init_db
    db = init_db()
    if db["ok"]:
        logger.info("PostgreSQL: tablas verificadas")
    else:
        logger.warning(f"PostgreSQL no disponible: {db['error']}")

    if settings.arcgis_configured:
        try:
            from tools.arcgis_tools import get_gis_connection
            gis = get_gis_connection()
            logger.info(f"ArcGIS conectado: {gis.properties.user.username}")
        except Exception as e:
            logger.warning(f"ArcGIS no disponible: {e}")
    else:
        logger.warning("ArcGIS sin credenciales — configurá ARCGIS_USERNAME y ARCGIS_PASSWORD")

    try:
        from llm.router import get_llm, get_llm_info
        get_llm()
        info = get_llm_info()
        logger.info(f"LLM activo: {info['provider']} — {info['model']}")
    except Exception as e:
        # Acá aparece el mensaje del router cuando falta, por ejemplo,
        # NVIDIA_NIM_API_KEY. Es informativo, no fatal.
        logger.warning(f"LLM no disponible: {e}")

    try:
        from llm.embeddings import get_embeddings
        emb = get_embeddings()
        logger.info(f"Embeddings: {emb.provider} — {emb.model}")
    except Exception as e:
        logger.warning(f"Embeddings no disponibles: {e}")

    logger.info(f"Sistema listo en http://0.0.0.0:{settings.api_port}")
    yield
    logger.info("Sistema apagado.")


app = FastAPI(
    title="Sistema Multi-Agente Geoespacial",
    description=(
        "Consultas geoespaciales en lenguaje natural sobre ArcGIS Online. "
        "LangGraph + LangChain + vectorización de WebMaps en ChromaDB, "
        "con LLM intercambiable (NVIDIA NIM, Ollama, Groq, OpenAI)."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router,    prefix="/api", tags=["chat"])
app.include_router(status_router,  prefix="/api", tags=["status"])
app.include_router(webmaps_router, prefix="/api", tags=["webmaps"])


@app.get("/health", tags=["status"])
async def health():
    """Healthcheck de Docker. No toca servicios externos a propósito."""
    return {"status": "ok", "version": "2.0.0"}


# ── Frontend compilado (opcional) ─────────────────────────
# En docker-compose el frontend se sirve desde su propio contenedor Bun.
# Este mount sirve para el modo "todo en un proceso": bun run build y listo.
_root = os.path.dirname(os.path.dirname(__file__))
_dist = os.path.join(_root, "frontend", "dist")

if os.path.isdir(_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_dist, "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(_dist, "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "sistema": "Multi-Agente Geoespacial",
            "version": "2.0.0",
            "docs"   : "/docs",
            "nota"   : "El frontend Svelte corre aparte en el puerto 5173.",
        }
