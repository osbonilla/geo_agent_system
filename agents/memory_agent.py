"""
agents/memory_agent.py
───────────────────────
MemoryAgent: persistencia y contexto histórico del sistema.

Tiene DOS modos de uso:

1. DIRECTO (sin LLM) — llamadas Python directas desde el Planner
   Usar para operaciones simples: guardar sesión, loggear acción

2. AGENTE (con LLM) — ReAct igual que el GeoAgent
   Usar cuando el usuario pregunta cosas como:
   "¿qué consulté antes?" o "muéstrame el historial"
"""
import logging
import json
from datetime import datetime
from sqlalchemy.orm import Session as DBSession
from db.connection import SessionLocal, test_connection
from db.models import Session, AgentLog, GeoResult, AgentRole
from tools.utils import generate_session_id
from tools.memory_tools import (
    get_recent_sessions,
    get_session_logs,
    search_sessions,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# El modelo solo LEE la memoria. Guardar sesiones y logs lo hace el código,
# nunca el LLM: un SLM no debería decidir qué se persiste.
MEMORY_TOOLS = [get_recent_sessions, get_session_logs, search_sessions]

MEMORY_SYSTEM = """Eres el MemoryAgent: consultas el historial de conversaciones guardado en PostgreSQL.
- últimas consultas → get_recent_sessions
- buscar por tema → search_sessions con una palabra clave
- detalle de una sesión → get_session_logs con su db_id
Responde en español, breve."""


# ══════════════════════════════════════════════════════════
# MODO 1: USO DIRECTO (sin LLM) — para el Planner
# ══════════════════════════════════════════════════════════

class MemoryAgent:
    """
    Uso directo desde Python — operaciones simples de DB.
    El Planner usa esta clase para guardar sesiones y logs.
    """

    def save_session(self, user_input: str) -> dict:
        """Crea y guarda una nueva sesión. Retorna session_id y db_id."""
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
            return {"session_id": session_id, "db_id": session.id}
        except Exception as e:
            db.rollback()
            logger.error(f"Error guardando sesión: {e}")
            return {"error": str(e)}
        finally:
            db.close()

    def log_agent_action(
        self,
        session_db_id: int,
        agent: AgentRole,
        action: str,
        input_data: dict = None,
        output_data: dict = None,
        status: str = "success",
        error_message: str = None,
        duration_ms: int = None,
    ) -> bool:
        """Registra una acción de agente. Retorna True si se guardó."""
        db: DBSession = SessionLocal()
        try:
            log = AgentLog(
                session_id=session_db_id,
                agent=agent,
                action=action,
                input_data=input_data,
                output_data=output_data,
                status=status,
                error_message=error_message,
                duration_ms=duration_ms,
                created_at=datetime.utcnow(),
            )
            db.add(log)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error guardando log: {e}")
            return False
        finally:
            db.close()

    def save_geo_result(
        self,
        session_db_id: int,
        layer_name: str,
        attributes: dict,
        geometry: dict = None,
        feature_id: str = None,
        source_url: str = None,
    ) -> bool:
        """Persiste un resultado geoespacial en PostGIS."""
        db: DBSession = SessionLocal()
        try:
            geom_wkt = self._arcgis_geom_to_wkt(geometry) if geometry else None
            result = GeoResult(
                session_id=session_db_id,
                layer_name=layer_name,
                feature_id=str(feature_id) if feature_id else None,
                attributes=attributes,
                geometry=f"SRID=4326;{geom_wkt}" if geom_wkt else None,
                source_url=source_url,
                created_at=datetime.utcnow(),
            )
            db.add(result)
            db.commit()
            logger.info(f"GeoResult guardado: layer={layer_name}")
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Error guardando GeoResult: {e}")
            return False
        finally:
            db.close()

    def get_recent_sessions(self, limit: int = 5) -> list:
        """Recupera las sesiones más recientes."""
        db: DBSession = SessionLocal()
        try:
            sessions = (
                db.query(Session)
                .order_by(Session.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "db_id"     : s.id,
                    "session_id": s.session_id,
                    "user_input": s.user_input,
                    "created_at": str(s.created_at),
                }
                for s in sessions
            ]
        except Exception as e:
            logger.error(f"Error recuperando sesiones: {e}")
            return []
        finally:
            db.close()

    def get_geo_results_by_session(self, session_db_id: int) -> list:
        """Recupera resultados geoespaciales de una sesión."""
        db: DBSession = SessionLocal()
        try:
            results = (
                db.query(GeoResult)
                .filter(GeoResult.session_id == session_db_id)
                .all()
            )
            return [
                {
                    "id"        : r.id,
                    "layer_name": r.layer_name,
                    "feature_id": r.feature_id,
                    "attributes": r.attributes,
                    "source_url": r.source_url,
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"Error recuperando GeoResults: {e}")
            return []
        finally:
            db.close()

    def check_db(self) -> dict:
        """Verifica el estado de la conexión a la base de datos."""
        return test_connection()

    def _arcgis_geom_to_wkt(self, geometry: dict) -> str:
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


# ══════════════════════════════════════════════════════════
# MODO 2: AGENTE CON LLM — para consultas del usuario
# ══════════════════════════════════════════════════════════

def create_memory_agent():
    from agents.base import build_agent
    return build_agent(MEMORY_SYSTEM, MEMORY_TOOLS, max_iterations=4)


def run_memory_agent(task: str) -> dict:
    from agents.base import run_agent
    return run_agent("MemoryAgent", create_memory_agent, task, grounding=False)
