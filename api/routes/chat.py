"""
api/routes/chat.py
───────────────────
POST /api/chat — el ruteo vive en agents/dispatch.py (lo comparten la API,
el CLI y la evaluación).

Es `def` y no `async def` a propósito: los agentes hacen llamadas bloqueantes
(LLM, ArcGIS, PostgreSQL). FastAPI corre las funciones `def` en un threadpool;
con `async def` congelarían el servidor entero.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

STEP_PREVIEW = 600   # caracteres de cada paso que viajan al frontend


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    webmap_id: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    answer: str
    agent_used: str
    steps_count: int
    steps: list
    session_id: Optional[str] = None
    duration_ms: int
    geojson: Optional[dict] = None
    sources: List[dict] = []
    map_action: Optional[dict] = None
    webmap_id: Optional[str] = None
    plan: List[dict] = []
    metrics: dict = {}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")

    session_db_id = None
    try:
        from agents.memory_agent import MemoryAgent
        session_db_id = MemoryAgent().save_session(message).get("db_id")
    except Exception as e:
        logger.warning(f"No se pudo guardar la sesión: {e}")

    from agents.dispatch import dispatch
    r = dispatch(message, webmap_id=request.webmap_id, session_db_id=session_db_id)

    steps_ui = [{**s, "input": str(s.get("input", ""))[:500],
                 "output": str(s.get("output", ""))[:STEP_PREVIEW]} for s in r["steps"]]

    return ChatResponse(
        success=r["success"], answer=r["answer"], agent_used=r["agent_used"],
        steps_count=r["steps_count"], steps=steps_ui,
        session_id=str(session_db_id) if session_db_id else None,
        duration_ms=r["metrics"]["duration_ms"], geojson=r["geojson"],
        sources=r["sources"], map_action=_map_action(r["geojson"], r["sources"]),
        webmap_id=request.webmap_id, plan=r["plan"], metrics=r["metrics"],
    )


def _map_action(geojson: Optional[dict], sources: List[dict]) -> Optional[dict]:
    if geojson and geojson.get("features"):
        return {"type": "draw_geojson", "features": len(geojson["features"])}
    if sources:
        return {"type": "add_layer", "url": sources[0]["url"], "title": sources[0]["capa"]}
    return None
