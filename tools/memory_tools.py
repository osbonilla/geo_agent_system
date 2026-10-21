"""
tools/memory_tools.py
──────────────────────
Tools de LangChain para el MemoryAgent.
Permiten guardar y recuperar información de PostgreSQL + PostGIS.

Herramientas:
  - save_session          → inicia y guarda una sesión de conversación
  - log_action            → registra acción de un agente
  - save_geo_result       → persiste resultado geoespacial en PostGIS
  - get_recent_sessions   → recupera historial de sesiones
  - get_session_logs      → recupera logs de una sesión específica
  - search_sessions       → busca sesiones por keyword
"""
import logging
import json
from datetime import datetime
from langchain_core.tools import tool
from sqlalchemy.orm import Session as DBSession
from db.connection import SessionLocal
from db.models import Session, AgentLog, GeoResult, AgentRole
from tools.utils import generate_session_id

logger = logging.getLogger(__name__)


# ── Helpers internos ───────────────────────────────────────

def _arcgis_geom_to_wkt(geometry: dict) -> str:
    """Convierte geometría ArcGIS JSON → WKT para PostGIS."""
    if not geometry:
        return None
    try:
        if "x" in geometry and "y" in geometry:
            return f"POINT({geometry['x']} {geometry['y']})"
        if "rings" in geometry:
            ring_str = ", ".join(f"{x} {y}" for x, y in geometry["rings"][0])
            return f"POLYGON(({ring_str}))"
        if "paths" in geometry:
            path_str = ", ".join(f"{x} {y}" for x, y in geometry["paths"][0])
            return f"LINESTRING({path_str})"
    except Exception as e:
        logger.warning(f"No se pudo convertir geometría: {e}")
    return None


# ── TOOLS ──────────────────────────────────────────────────

@tool
def save_session(user_input: str) -> str:
    """
    Crea y guarda una nueva sesión de conversación en PostgreSQL.
    Llamar al inicio de cada conversación del usuario.
    Action Input: el texto de la pregunta o tarea del usuario
    Retorna el session_id y db_id generados.
    """
    db: DBSession = SessionLocal()
    try:
        session_id = generate_session_id()
        session = Session(
            session_id=session_id,
            user_input=str(user_input).strip(),
            created_at=datetime.utcnow(),
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        logger.info(f"Sesión guardada: {session_id}")
        return json.dumps({
            "status": "guardado",
            "session_id": session_id,
            "db_id": session.id,
            "user_input": user_input,
        }, ensure_ascii=False)
    except Exception as e:
        db.rollback()
        logger.error(f"Error guardando sesión: {e}")
        return json.dumps({"status": "error", "mensaje": str(e)})
    finally:
        db.close()


@tool
def log_action(input_str: str) -> str:
    """
    Registra la acción de un agente en la base de datos.
    Action Input: JSON con campos: session_db_id, agent, action, output, status
    Ejemplo: {"session_db_id": 1, "agent": "GEO_AGENT", "action": "list_content", "output": "...", "status": "success"}
    Agentes válidos: PLANNER, GEO_AGENT, QUERY_AGENT, MEMORY_AGENT
    """
    db: DBSession = SessionLocal()
    try:
        if isinstance(input_str, dict):
            params = input_str
        else:
            params = json.loads(str(input_str))

        agent_str = params.get("agent", "GEO_AGENT").upper()
        agent_role = AgentRole[agent_str]

        log = AgentLog(
            session_id=int(params["session_db_id"]),
            agent=agent_role,
            action=str(params.get("action", "unknown")),
            input_data=params.get("input_data"),
            output_data={"output": params.get("output", "")},
            status=params.get("status", "success"),
            error_message=params.get("error_message"),
            duration_ms=params.get("duration_ms"),
            created_at=datetime.utcnow(),
        )
        db.add(log)
        db.commit()
        return json.dumps({"status": "guardado", "log_id": log.id}, ensure_ascii=False)
    except Exception as e:
        db.rollback()
        logger.error(f"Error guardando log: {e}")
        return json.dumps({"status": "error", "mensaje": str(e)})
    finally:
        db.close()


@tool
def save_geo_result(input_str: str) -> str:
    """
    Persiste un resultado geoespacial en PostGIS.
    Action Input: JSON con campos: session_db_id, layer_name, attributes, geometry (opcional), source_url (opcional)
    Ejemplo: {"session_db_id": 1, "layer_name": "Arbolado Quito", "attributes": {"especie": "Eucalipto"}, "geometry": {"x": -78.5, "y": -0.2}}
    La geometría se convierte automáticamente a PostGIS (SRID 4326).
    """
    db: DBSession = SessionLocal()
    try:
        if isinstance(input_str, dict):
            params = input_str
        else:
            params = json.loads(str(input_str))

        geometry = params.get("geometry")
        geom_wkt = _arcgis_geom_to_wkt(geometry) if geometry else None

        result = GeoResult(
            session_id=int(params["session_db_id"]),
            layer_name=str(params.get("layer_name", "unknown")),
            feature_id=str(params.get("feature_id", "")),
            attributes=params.get("attributes", {}),
            geometry=f"SRID=4326;{geom_wkt}" if geom_wkt else None,
            source_url=params.get("source_url"),
            created_at=datetime.utcnow(),
        )
        db.add(result)
        db.commit()
        logger.info(f"GeoResult guardado: {params.get('layer_name')}")
        return json.dumps({
            "status": "guardado",
            "geo_result_id": result.id,
            "layer_name": params.get("layer_name"),
            "tiene_geometria": geom_wkt is not None,
        }, ensure_ascii=False)
    except Exception as e:
        db.rollback()
        logger.error(f"Error guardando GeoResult: {e}")
        return json.dumps({"status": "error", "mensaje": str(e)})
    finally:
        db.close()


@tool
def get_recent_sessions(limit: int = 5) -> str:
    """
    Recupera las sesiones de conversación más recientes.
    Útil para dar contexto histórico al usuario o al Planner.
    Action Input: número de sesiones a recuperar, ejemplo: 5
    Retorna lista de sesiones con su pregunta original y fecha.
    """
    db: DBSession = SessionLocal()
    try:
        if isinstance(limit, str):
            import re
            nums = re.findall(r'\d+', limit)
            limit = int(nums[0]) if nums else 5

        sessions = (
            db.query(Session)
            .order_by(Session.created_at.desc())
            .limit(int(limit))
            .all()
        )
        result = []
        for s in sessions:
            # Contar logs y geo_results asociados
            logs_count = db.query(AgentLog).filter(AgentLog.session_id == s.id).count()
            geo_count  = db.query(GeoResult).filter(GeoResult.session_id == s.id).count()
            result.append({
                "db_id"      : s.id,
                "session_id" : s.session_id,
                "user_input" : s.user_input,
                "created_at" : str(s.created_at),
                "logs"       : logs_count,
                "geo_results": geo_count,
            })
        return json.dumps({
            "total": len(result),
            "sesiones": result,
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error recuperando sesiones: {e}")
        return json.dumps({"status": "error", "mensaje": str(e)})
    finally:
        db.close()


@tool
def get_session_logs(session_db_id: int) -> str:
    """
    Recupera todos los logs de acciones de una sesión específica.
    Muestra qué agentes actuaron y qué hicieron en esa sesión.
    Action Input: el db_id numérico de la sesión, ejemplo: 1
    """
    db: DBSession = SessionLocal()
    try:
        if isinstance(session_db_id, str):
            import re
            nums = re.findall(r'\d+', session_db_id)
            session_db_id = int(nums[0]) if nums else 0

        logs = (
            db.query(AgentLog)
            .filter(AgentLog.session_id == int(session_db_id))
            .order_by(AgentLog.created_at.asc())
            .all()
        )
        result = []
        for log in logs:
            result.append({
                "agente"   : log.agent.value if log.agent else "N/A",
                "accion"   : log.action,
                "status"   : log.status,
                "output"   : str(log.output_data)[:200] if log.output_data else "",
                "fecha"    : str(log.created_at),
            })
        return json.dumps({
            "session_db_id": session_db_id,
            "total_logs": len(result),
            "logs": result,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "mensaje": str(e)})
    finally:
        db.close()


@tool
def search_sessions(keyword: str) -> str:
    """
    Busca sesiones anteriores que contengan una palabra clave.
    Útil para saber si el usuario ya preguntó algo similar antes.
    Action Input: la palabra clave a buscar, ejemplo: arbolado
    """
    db: DBSession = SessionLocal()
    try:
        keyword = str(keyword).strip()
        sessions = (
            db.query(Session)
            .filter(Session.user_input.ilike(f"%{keyword}%"))
            .order_by(Session.created_at.desc())
            .limit(10)
            .all()
        )
        result = [
            {
                "db_id"     : s.id,
                "session_id": s.session_id,
                "user_input": s.user_input,
                "created_at": str(s.created_at),
            }
            for s in sessions
        ]
        return json.dumps({
            "busqueda": keyword,
            "total": len(result),
            "sesiones": result,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "mensaje": str(e)})
    finally:
        db.close()
