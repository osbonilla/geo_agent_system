"""
agents/geo_agent.py
────────────────────
GeoAgent: encuentra capas en ArcGIS Online y consulta sus registros,
incluidas consultas espaciales por distancia. Tool calling nativo.
"""
from typing import Optional

from agents.base import build_agent, run_agent
from tools.arcgis_tools import GEO_TOOLS

GEO_SYSTEM = """Eres el GeoAgent, especialista en ArcGIS Online con datos de Ecuador.

Cómo trabajar:
1. Si te dan CAPAS CANDIDATAS con URL, úsalas directamente. Si no, busca la capa con search_layers.
2. Antes de filtrar por un campo, confirma su nombre con get_layer_info.
3. Para contar usa count_features; para ver registros, query_features;
   para "cerca de", "a menos de X metros", "alrededor de", usa query_features_near.
4. Los filtros where son SQL: texto entre comillas simples, ej. especie = 'Eucalipto'.
5. Si una herramienta devuelve error, lee el mensaje (trae los campos válidos) y corrige.

Responde en español, breve, con los números exactos que devolvieron las herramientas.
No inventes URLs, campos ni cifras."""


def create_geo_agent():
    return build_agent(GEO_SYSTEM, GEO_TOOLS)


def run_geo_agent(task: str, webmap_id: Optional[str] = None,
        grounding_query: Optional[str] = None) -> dict:
    return run_agent("GeoAgent", create_geo_agent, task, webmap_id=webmap_id,
                     grounding_query=grounding_query)


def run_geo_agent_simple(task: str) -> str:
    return run_geo_agent(task)["output"]
