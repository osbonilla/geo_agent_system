"""
db/models.py
────────────
Modelos ORM del sistema multi-agente geoespacial.

Tablas:
  - sessions       → conversaciones con el usuario
  - agent_logs     → registro de decisiones del planner y agentes
  - geo_results    → resultados geoespaciales persistidos (PostGIS)
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime,
    ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from db.connection import Base
import enum


class AgentRole(str, enum.Enum):
    PLANNER = "planner"
    GEO_AGENT = "geo_agent"
    QUERY_AGENT = "query_agent"
    MEMORY_AGENT = "memory_agent"
    ANALYTICS_AGENT = "analytics_agent"
    RAG_AGENT = "rag_agent"
    SPATIAL_AGENT = "spatial_agent"


class Session(Base):
    """
    Representa una conversación completa con el usuario.
    Un usuario puede tener múltiples sesiones.
    """
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    user_input = Column(Text, nullable=False)          # Primera pregunta
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relaciones
    logs = relationship("AgentLog", back_populates="session", cascade="all, delete")
    geo_results = relationship("GeoResult", back_populates="session", cascade="all, delete")

    def __repr__(self):
        return f"<Session id={self.id} session_id={self.session_id}>"


class AgentLog(Base):
    """
    Registro de cada acción ejecutada por un agente.
    Clave para auditoría y debug de la tesis.
    """
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    agent = Column(SAEnum(AgentRole), nullable=False)
    action = Column(String(128), nullable=False)       # nombre de la tool usada
    input_data = Column(JSON, nullable=True)           # parámetros de entrada
    output_data = Column(JSON, nullable=True)          # resultado
    status = Column(String(16), default="success")     # success | error
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)       # tiempo de ejecución
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="logs")

    def __repr__(self):
        return f"<AgentLog agent={self.agent} action={self.action}>"


class GeoResult(Base):
    """
    Resultados geoespaciales persistidos.
    La columna 'geometry' usa PostGIS — acepta cualquier tipo de geometría.
    """
    __tablename__ = "geo_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    layer_name = Column(String(256), nullable=False)   # nombre de la capa ArcGIS
    feature_id = Column(String(128), nullable=True)    # OBJECTID del feature
    attributes = Column(JSON, nullable=True)           # atributos del feature
    geometry = Column(
        Geometry(geometry_type="GEOMETRY", srid=4326),
        nullable=True
    )                                                  # geometría real en WGS84
    source_url = Column(Text, nullable=True)           # URL del servicio ArcGIS
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="geo_results")

    def __repr__(self):
        return f"<GeoResult layer={self.layer_name} feature={self.feature_id}>"


def _sync_agentrole_enum(engine) -> None:
    """
    create_all no modifica un tipo ENUM que ya existe en PostgreSQL. Si la base se
    creó antes de agregar un rol (p. ej. SPATIAL_AGENT), insertar ese log fallaría.
    Se agregan los valores que falten; SQLAlchemy guarda el NOMBRE del miembro.
    ALTER TYPE ... ADD VALUE va fuera de una transacción (AUTOCOMMIT).
    """
    from sqlalchemy import text
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        existe = conn.execute(text("SELECT 1 FROM pg_type WHERE typname = 'agentrole'")).first()
        if not existe:
            return
        for rol in AgentRole:
            conn.execute(text(f"ALTER TYPE agentrole ADD VALUE IF NOT EXISTS '{rol.name}'"))


def init_db() -> dict:
    """
    Crea la extensión PostGIS y las tablas si no existen.
    Se llama al arrancar la API y el CLI: sin esto, en una base nueva
    no existe la tabla `sessions` y toda la memoria falla en silencio.
    """
    from sqlalchemy import text
    from db.connection import engine
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        Base.metadata.create_all(bind=engine)
        _sync_agentrole_enum(engine)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
