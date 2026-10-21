"""
agents/analytics_agent.py
──────────────────────────
AnalyticsAgent: estadísticas sobre Feature Layers de ArcGIS Online.
El servidor de ArcGIS calcula; el modelo solo elige capa, campo y operación.
"""
from typing import Optional

from agents.base import build_agent, run_agent
from tools.analytics_tools import ANALYTICS_TOOLS
from tools.arcgis_tools import search_layers, get_layer_info

ANALYTICS_SYSTEM = """Eres el AnalyticsAgent, especialista en estadísticas de capas de ArcGIS Online (Ecuador).

Cómo trabajar:
1. Necesitas la URL de la capa. Úsala de las CAPAS CANDIDATAS o del contexto;
   si no la tienes, búscala con search_layers.
2. Elige UNA herramienta según la pregunta:
   - máximo, mínimo, promedio, suma de un campo → field_statistics
   - los N más altos / más bajos → top_n_features
   - cuántos hay por categoría, distribución → count_by_category
   - analiza / resume la capa → layer_summary
3. Si no sabes el nombre del campo, consulta get_layer_info. Puedes pasar el alias.
4. Si una herramienta devuelve error, lee el mensaje (trae los campos válidos) y corrige.

Responde en español, breve, citando los números exactos. Nunca calcules tú: usa los
valores que devuelven las herramientas."""

TOOLS = ANALYTICS_TOOLS + [search_layers, get_layer_info]


def create_analytics_agent():
    return build_agent(ANALYTICS_SYSTEM, TOOLS)


def run_analytics_agent(task: str, webmap_id: Optional[str] = None,
        grounding_query: Optional[str] = None) -> dict:
    return run_agent("AnalyticsAgent", create_analytics_agent, task, webmap_id=webmap_id,
                     grounding_query=grounding_query)


def run_analytics_agent_simple(task: str) -> str:
    return run_analytics_agent(task)["output"]
