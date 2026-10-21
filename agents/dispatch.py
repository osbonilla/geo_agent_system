"""
agents/dispatch.py
──────────────────
Punto de entrada único: recibe una pregunta y decide qué agente la resuelve.

Lo usan la API (/api/chat), el CLI y la evaluación (eval/run_eval.py), así
que las tres miden exactamente el mismo sistema.

Ruteo
─────
  "geo: ..."        → GeoAgent
  "query: ..."      → AnalyticsAgent  (estadísticas sobre ArcGIS)
  "analytics: ..."  → AnalyticsAgent
  "spatial: ..."    → SpatialAgent    (relaciones espaciales entre capas)
  "postgis: ..."    → QueryAgent      (PostGIS local)
  "rag: ..."        → RAGAgent
  "plan: ..."       → Planner
  sin prefijo       → Planner (si hay un WebMap abierto, el Planner recibe sus capas
                      y decide: describir el mapa → RAGAgent; pedir datos de una
                      capa → AnalyticsAgent / GeoAgent; cruzar capas → SpatialAgent)

Al final de cada petición:
  · las features que trajeron las herramientas se devuelven como GeoJSON
  · se guardan en PostGIS (geo_results) para el QueryAgent
  · se miden llamadas y tokens del LLM
"""
import logging
import time
from typing import Optional, List

from llm.metrics import COUNTER
from tools import result_store

logger = logging.getLogger(__name__)

PREFIXES = [
    ("geo:", "GeoAgent"), ("query:", "AnalyticsAgent"), ("analytics:", "AnalyticsAgent"),
    ("spatial:", "SpatialAgent"),
    ("postgis:", "QueryAgent"), ("rag:", "RAGAgent"), ("plan:", "Planner"),
]
MAX_PERSIST = 200


def route(message: str, webmap_id: Optional[str]) -> tuple:
    low = message.lower()
    for prefix, agent in PREFIXES:
        if low.startswith(prefix):
            return agent, message[len(prefix):].strip()
    return "Planner", message


def _run(agent: str, task: str, webmap_id: Optional[str], session_db_id: Optional[int]) -> dict:
    if agent == "GeoAgent":
        from agents.geo_agent import run_geo_agent
        return run_geo_agent(task, webmap_id=webmap_id)
    if agent == "AnalyticsAgent":
        from agents.analytics_agent import run_analytics_agent
        return run_analytics_agent(task, webmap_id=webmap_id)
    if agent == "SpatialAgent":
        from agents.spatial_agent import run_spatial_agent
        return run_spatial_agent(task, webmap_id=webmap_id)
    if agent == "QueryAgent":
        from agents.query_agent import run_query_agent
        return run_query_agent(task)
    if agent == "RAGAgent":
        from agents.rag_agent import run_rag_agent
        return run_rag_agent(task, webmap_id=webmap_id)

    from agents.planner import run_planner
    r = run_planner(task, session_db_id=session_db_id, webmap_id=webmap_id)
    return {
        "success": r["success"],
        "output": r["final_answer"],
        "steps": [{"tool": f"{s.get('agente', '')} · {s.get('tool', '')}",
                   "input": s.get("input", ""), "output": s.get("output", "")}
                  for s in r.get("steps", [])],
        "steps_count": r.get("steps_count", 0),
        "sources": r.get("sources", []),
        "plan": [{"agente": a, "tarea": t} for a, t in zip(r.get("agentes", []), r.get("plan", []))],
    }


def to_geojson(items: List[dict]) -> Optional[dict]:
    feats = []
    for it in items:
        g = it.get("geometry") or {}
        if "x" in g and "y" in g:
            geom = {"type": "Point", "coordinates": [g["x"], g["y"]]}
        elif "rings" in g:
            geom = {"type": "Polygon", "coordinates": g["rings"]}
        elif "paths" in g:
            geom = {"type": "MultiLineString", "coordinates": g["paths"]}
        else:
            continue
        feats.append({"type": "Feature", "geometry": geom,
                      "properties": {**(it.get("attributes") or {}), "_capa": it.get("layer_name", "")}})
    return {"type": "FeatureCollection", "features": feats} if feats else None


def _persist(items: List[dict], session_db_id: Optional[int]) -> int:
    if not session_db_id:
        return 0
    from agents.memory_agent import MemoryAgent
    mem, n = MemoryAgent(), 0
    for it in items[:MAX_PERSIST]:
        if it.get("layer_url", "").startswith("postgis://"):
            continue  # ya viene de PostGIS
        attrs = it.get("attributes") or {}
        fid = attrs.get("OBJECTID", attrs.get("objectid", attrs.get("FID")))
        if mem.save_geo_result(session_db_id, it.get("layer_name") or "capa", attrs,
                               it.get("geometry"), feature_id=fid, source_url=it.get("layer_url")):
            n += 1
    return n


def dispatch(message: str, webmap_id: Optional[str] = None,
             session_db_id: Optional[int] = None, persist: bool = True) -> dict:
    """
    Returns: success, answer, agent_used, steps, steps_count, sources, plan,
             geojson, metrics {llm_calls, prompt_tokens, completion_tokens, duration_ms}
    """
    COUNTER.reset()
    result_store.reset()
    start = time.time()

    agent, task = route(message.strip(), webmap_id)
    try:
        result = _run(agent, task, webmap_id, session_db_id)
    except Exception as e:
        logger.exception("Error ejecutando agente")
        result = {"success": False, "output": f"Error: {e}", "steps": [], "steps_count": 0}

    items = result_store.items()
    persisted = _persist(items, session_db_id) if persist else 0

    return {
        "success": result.get("success", False),
        "answer": result.get("output", "Sin respuesta"),
        "agent_used": agent,
        "steps": result.get("steps", []),
        "steps_count": result.get("steps_count", 0),
        "sources": result.get("sources", []),
        "plan": result.get("plan", []),
        "geojson": to_geojson(items),
        "metrics": {**COUNTER.snapshot(), "duration_ms": int((time.time() - start) * 1000),
                    "features": len(items), "persistidas": persisted,
                    "grounding": result.get("grounding")},
    }
