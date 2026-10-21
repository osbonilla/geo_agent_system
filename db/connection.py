"""
db/connection.py
────────────────
Pool de conexión a PostgreSQL + PostGIS.
Usa SQLAlchemy con soporte GeoAlchemy2 para geometrías.
"""
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import NullPool
from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Motor principal — NullPool evita problemas en entornos async
engine = create_engine(
    settings.database_url,
    poolclass=NullPool,
    echo=False,  # poné True para ver el SQL en consola
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base para todos los modelos ORM del proyecto."""
    pass


def get_db():
    """
    Generador de sesión — usar con 'with' o en FastAPI como dependency.
    Garantiza cierre de sesión aunque ocurra un error.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection() -> dict:
    """
    Verifica la conexión a PostgreSQL y que PostGIS esté instalado.
    Retorna un dict con el estado y la versión de PostGIS.
    """
    try:
        with engine.connect() as conn:
            # Verificar PostgreSQL
            pg_version = conn.execute(text("SELECT version()")).scalar()

            # Verificar PostGIS
            postgis_version = conn.execute(
                text("SELECT PostGIS_Version()")
            ).scalar()

        return {
            "status": "ok",
            "postgresql": pg_version.split(",")[0],
            "postgis": postgis_version,
        }
    except Exception as e:
        logger.error(f"Error de conexión a la base de datos: {e}")
        return {"status": "error", "message": str(e)}
