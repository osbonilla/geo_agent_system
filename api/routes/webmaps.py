"""
api/routes/webmaps.py
─────────────────────
WebMaps de ArcGIS Online y su índice vectorial.

  GET    /api/webmaps              → lista los WebMaps de la organización
  GET    /api/webmaps/{id}         → ficha + capas de un WebMap
  POST   /api/ingest-webmap        → vectoriza un WebMap (o todos)
  DELETE /api/webmaps/{id}/index   → borra el índice de un WebMap
  POST   /api/webmaps/search       → búsqueda semántica sobre el índice
  GET    /api/vector-status        → estado de ChromaDB
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


def _corto(e: Exception, n: int = 400) -> str:
    """Mensaje de error legible: algunas excepciones traen vectores enteros."""
    msg = str(e)
    return msg if len(msg) <= n else msg[:n] + " …"


class IngestRequest(BaseModel):
    webmap_id: Optional[str] = None   # vacío = vectorizar todos
    deep: bool = True                 # consultar campos y conteos reales
    max_items: int = 20               # tope cuando se vectoriza todo


class SearchRequest(BaseModel):
    query: str
    webmap_id: Optional[str] = None
    k: int = 5


@router.get("/webmaps")
def get_webmaps(max_items: int = 50, q: str = None):
    """Lista los WebMaps disponibles en ArcGIS Online."""
    try:
        from tools.vector_tools import list_webmaps, vector_status

        webmaps = list_webmaps(max_items=max_items, query=q)

        # Marcar cuáles ya están vectorizados
        indexados = {
            m.get("webmap_id"): m
            for m in vector_status().get("webmaps", [])
        }
        for wm in webmaps:
            meta = indexados.get(wm["id"])
            wm["vectorizado"] = meta is not None
            wm["embedding"] = (meta or {}).get("embedding", "")

        return {"total": len(webmaps), "webmaps": webmaps}
    except Exception as e:
        logger.error(f"Error listando WebMaps: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/webmaps/{webmap_id}")
def get_webmap(webmap_id: str):
    """Ficha completa de un WebMap: metadatos, capas y campos."""
    try:
        from tools.vector_tools import read_webmap
        return read_webmap(webmap_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/ingest-webmap")
def ingest_webmap(request: IngestRequest):
    """
    Vectoriza un WebMap y lo deja consultable por el RAGAgent.
    Sin webmap_id vectoriza todos los WebMaps del usuario.
    """
    try:
        from tools.vector_tools import vectorize_webmap, vectorize_all_webmaps

        if request.webmap_id:
            return vectorize_webmap(request.webmap_id, deep=request.deep)
        return vectorize_all_webmaps(max_items=request.max_items, deep=request.deep)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error vectorizando: {_corto(e, 2000)}")
        raise HTTPException(status_code=500, detail=_corto(e))


@router.delete("/webmaps/{webmap_id}/index")
def delete_index(webmap_id: str):
    """Borra el índice vectorial de un WebMap para regenerarlo desde cero."""
    from tools.vector_tools import delete_webmap_index
    result = delete_webmap_index(webmap_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error", "no encontrado"))
    return result


@router.post("/webmaps/search")
def search_index(request: SearchRequest):
    """Búsqueda semántica directa sobre el índice, sin pasar por el LLM."""
    try:
        from tools.vector_tools import search_webmap
        hits = search_webmap(request.query, webmap_id=request.webmap_id, k=request.k)
        return {"total": len(hits), "resultados": hits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector-status")
def get_vector_status():
    """Estado de ChromaDB y del backend de embeddings."""
    from tools.vector_tools import vector_status
    from llm.embeddings import get_embeddings_info

    return {"chroma": vector_status(), "embeddings": get_embeddings_info()}
