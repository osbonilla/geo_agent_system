"""
api/routes/status.py
─────────────────────
Estado del sistema y control del LLM.

  GET  /api/status        → ArcGIS · LLM · PostgreSQL · ChromaDB
  GET  /api/llm-info      → modelo activo
  GET  /api/llm-providers → proveedores soportados y cuáles tienen key
  POST /api/llm-select    → cambiar de modelo en caliente
  POST /api/llm-reset     → re-detectar
  GET  /api/sessions      → historial de sesiones
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class SystemStatus(BaseModel):
    arcgis: dict
    llm: dict
    database: dict
    vectors: dict


class SelectModelRequest(BaseModel):
    model: str   # formato <proveedor>/<modelo>, ej. nvidia_nim/openai/gpt-oss-20b


@router.get("/status", response_model=SystemStatus)
def get_status():
    """Verifica todas las conexiones del sistema de una sola vez."""

    # ArcGIS
    try:
        from config.settings import get_settings
        settings = get_settings()
        if not settings.arcgis_configured:
            arcgis_status = {
                "ok": False,
                "error": "Falta ARCGIS_USERNAME / ARCGIS_PASSWORD en el .env",
            }
        else:
            from tools.arcgis_tools import get_gis_connection
            gis = get_gis_connection()
            arcgis_status = {
                "ok"     : True,
                "usuario": gis.properties.user.username,
                "url"    : settings.arcgis_url,
            }
    except Exception as e:
        arcgis_status = {"ok": False, "error": str(e)}

    # LLM
    try:
        from llm.router import get_llm_info
        llm_status = get_llm_info()
    except Exception as e:
        llm_status = {"ok": False, "error": str(e)}

    # PostgreSQL
    try:
        from db.connection import test_connection
        db_result = test_connection()
        db_status = {
            "ok"      : db_result["status"] == "ok",
            "postgres": db_result.get("postgresql", ""),
            "postgis" : db_result.get("postgis", ""),
            "error"   : db_result.get("message", ""),
        }
    except Exception as e:
        db_status = {"ok": False, "error": str(e)}

    # ChromaDB
    try:
        from tools.vector_tools import vector_status
        from llm.embeddings import get_embeddings_info
        vec = vector_status()
        vec["embeddings"] = get_embeddings_info()
    except Exception as e:
        vec = {"ok": False, "error": str(e)}

    return SystemStatus(
        arcgis  = arcgis_status,
        llm     = llm_status,
        database= db_status,
        vectors = vec,
    )


@router.get("/llm-info")
def llm_info():
    """Modelo activo: proveedor, nombre y si corre local."""
    from llm.router import get_llm_info
    return get_llm_info()


@router.get("/llm-providers")
def llm_providers():
    """Proveedores soportados, con la variable de entorno que necesita cada uno."""
    from llm.router import available_providers
    return {"providers": available_providers()}


@router.get("/ollama-models")
def ollama_models():
    """
    Modelos de chat instalados en Ollama (ollama list), para el selector de la interfaz.
    Se excluyen los de embeddings. Si Ollama no responde, lista vacía.
    """
    import httpx
    from config.settings import get_settings
    try:
        r = httpx.get(f"{get_settings().ollama_base_url}/api/tags", timeout=3.0)
        r.raise_for_status()
        nombres = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return {"models": []}
    embeddings = ("embed", "bge", "minilm", "e5-", "gte")
    return {"models": sorted(f"ollama/{n}" for n in nombres
                             if n and not any(e in n.lower() for e in embeddings))}


@router.post("/llm-select")
def llm_select(request: SelectModelRequest):
    """
    Cambia el modelo activo sin reiniciar el servidor.
    Body: {"model": "nvidia_nim/openai/gpt-oss-20b"}
    El cambio dura lo que dure el proceso; para que sea permanente, editá LLM_MODEL.
    """
    from llm.router import set_llm
    try:
        return {"ok": True, **set_llm(request.model)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm-reset")
def llm_reset():
    """Re-detecta el proveedor. Útil si Ollama arrancó después que la API."""
    from llm.router import reset_router
    return {"ok": True, "llm": reset_router()}


@router.get("/sessions")
def get_sessions(limit: int = 10):
    """Historial de sesiones recientes guardadas por el MemoryAgent."""
    try:
        from agents.memory_agent import MemoryAgent
        mem = MemoryAgent()
        sessions = mem.get_recent_sessions(limit=limit)
        return {"total": len(sessions), "sessions": sessions}
    except Exception as e:
        return {"total": 0, "sessions": [], "error": str(e)}
