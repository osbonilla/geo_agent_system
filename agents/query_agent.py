"""
agents/query_agent.py
──────────────────────
QueryAgent: consultas espaciales sobre PostGIS local.

  GeoAgent   → datos en ArcGIS Online (en vivo)
  QueryAgent → registros que ya se trajeron de ArcGIS durante las
               conversaciones y quedaron guardados en PostGIS
"""
from agents.base import build_agent, run_agent
from tools.postgis_tools import QUERY_TOOLS

QUERY_SYSTEM = """Eres el QueryAgent, especialista en consultas espaciales sobre PostgreSQL + PostGIS.
Trabajas con registros que ya se consultaron antes en ArcGIS Online y quedaron guardados.

Herramientas:
- count_by_layer: qué hay guardado y cuánto
- query_geo_results: ver registros guardados
- spatial_buffer: registros a menos de N metros de un punto
- find_nearest: el registro más cercano a un punto
- run_sql_spatial: SQL de solo lectura, solo si nada más sirve

Coordenadas en WGS84 (lon, lat). Quito centro: lon=-78.4678, lat=-0.1807. Distancias en metros.
Responde en español, breve, con los números exactos que devuelven las herramientas."""


def create_query_agent():
    return build_agent(QUERY_SYSTEM, QUERY_TOOLS)


def run_query_agent(task: str) -> dict:
    return run_agent("QueryAgent", create_query_agent, task, grounding=False)


def run_query_agent_simple(task: str) -> str:
    return run_query_agent(task)["output"]
