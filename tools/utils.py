"""
tools/utils.py
──────────────
Utilidades compartidas entre tools y agentes.
"""
import uuid
import logging
from datetime import datetime


def generate_session_id() -> str:
    """Genera un ID único de sesión para rastrear conversaciones."""
    return str(uuid.uuid4())


def setup_logging(level: str = "INFO"):
    """Configura el logging del sistema con formato legible."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def timestamp_now() -> str:
    """Retorna timestamp actual en formato ISO."""
    return datetime.utcnow().isoformat()
