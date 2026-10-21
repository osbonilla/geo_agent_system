"""
agents/planner.py
──────────────────
PlannerAgent: orquestador del sistema multi-agente con LangGraph.

Patrón Plan & Execute:
  1. PLAN    → el LLM genera una lista de tareas, cada una con su agente
  2. EXECUTE → cada tarea se ejecuta con el agente asignado; recibe como
               contexto las salidas de los pasos anteriores
  3. REPLAN  → el LLM decide si alcanza o si agrega otra tarea

Grafo:  planner → execute (bucle) → replan → (execute | END)

Notas de diseño
───────────────
· Toda clave que un nodo devuelve DEBE estar declarada en PlannerState.
  LangGraph descarta en silencio las claves que no están en el esquema
  (la versión anterior guardaba "_agentes" fuera del esquema y por eso
  todas las tareas terminaban ejecutándose con GeoAgent).
· Los nodos devuelven solo las claves que cambian (actualizaciones parciales).
"""
import json
import logging
import re
from typing import TypedDict, List, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, field_validator

from llm.structured import invoke_structured
from agents.memory_agent import MemoryAgent
from db.models import AgentRole

logger = logging.getLogger(__name__)

MAX_TAREAS = 6          # tope duro del plan, incluido el replan
CTX_POR_PASO = 1500     # caracteres de cada salida previa que ve el siguiente paso


# ══════════════════════════════════════════════════════════
# ESTADO DEL GRAFO
# ══════════════════════════════════════════════════════════

class PlannerState(TypedDict):
    user_input    : str
    webmap_id     : Optional[str]
    plan          : List[str]     # descripción de cada tarea
    agentes       : List[str]     # agente asignado a cada tarea (mismo índice)
    current_step  : int
    results       : List[dict]    # un resultado por tarea ejecutada
    steps         : List[dict]    # pasos de herramientas de todos los agentes
    sources       : List[dict]    # capas citadas (las devuelve el RAGAgent)
    final_answer  : str
    should_end    : bool
    session_db_id : int


# ══════════════════════════════════════════════════════════
# AGENTES DISPONIBLES
# ══════════════════════════════════════════════════════════

def _gq(tarea: str, state: PlannerState) -> str:
    """Consulta de grounding: pregunta original + subtarea, SIN el contexto de pasos previos."""
    sub = tarea.split("\n\nCONTEXTO DE PASOS ANTERIORES")[0]
    return f"{state['user_input']} {sub}"


def _run_geo(tarea: str, state: PlannerState) -> dict:
    from agents.geo_agent import run_geo_agent
    return run_geo_agent(tarea, webmap_id=state.get("webmap_id"),
                         grounding_query=_gq(tarea, state))


def _run_analytics(tarea: str, state: PlannerState) -> dict:
    from agents.analytics_agent import run_analytics_agent
    return run_analytics_agent(tarea, webmap_id=state.get("webmap_id"),
                               grounding_query=_gq(tarea, state))


def _run_spatial(tarea: str, state: PlannerState) -> dict:
    from agents.spatial_agent import run_spatial_agent
    return run_spatial_agent(tarea, webmap_id=state.get("webmap_id"),
                             grounding_query=_gq(tarea, state))


def _run_query(tarea: str, state: PlannerState) -> dict:
    from agents.query_agent import run_query_agent
    return run_query_agent(tarea)


def _run_rag(tarea: str, state: PlannerState) -> dict:
    from agents.rag_agent import run_rag_agent
    # Si es la única tarea, se busca con la pregunta original del usuario:
    # la reformulación del Planner suele perder matices que sí sirven al retrieval.
    consulta = state["user_input"] if len(state["plan"]) == 1 else tarea
    return run_rag_agent(consulta, webmap_id=state.get("webmap_id"))


def _run_memory(tarea: str, state: PlannerState) -> dict:
    sessions = MemoryAgent().get_recent_sessions(limit=5)
    out = json.dumps({"sesiones_recientes": sessions}, ensure_ascii=False, default=str)
    return {"success": True, "output": out,
            "steps": [{"tool": "get_recent_sessions", "input": "5", "output": out}],
            "steps_count": 1}


AGENTS = {
    "GeoAgent"      : _run_geo,
    "AnalyticsAgent": _run_analytics,
    "SpatialAgent"  : _run_spatial,
    "QueryAgent"    : _run_query,
    "RAGAgent"      : _run_rag,
    "MemoryAgent"   : _run_memory,
}

AGENT_ROLES = {
    "GeoAgent"      : AgentRole.GEO_AGENT,
    "AnalyticsAgent": AgentRole.ANALYTICS_AGENT,
    "SpatialAgent"  : AgentRole.SPATIAL_AGENT,
    "QueryAgent"    : AgentRole.QUERY_AGENT,
    "RAGAgent"      : AgentRole.RAG_AGENT,
    "MemoryAgent"   : AgentRole.MEMORY_AGENT,
}


def _normalize_agent(name: str) -> str:
    """Tolera 'geoagent', 'Geo Agent', 'analytics' … que suelen devolver los SLM."""
    key = re.sub(r"[^a-z]", "", str(name).lower())
    for canonical in AGENTS:
        c = canonical.lower()
        if key == c or key == c.replace("agent", ""):
            return canonical
    logger.warning(f"Agente desconocido '{name}' → GeoAgent")
    return "GeoAgent"


def _parse_json(content: str) -> dict:
    """Extrae el primer objeto JSON de la respuesta, con o sin ```json```."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        raise ValueError(f"La respuesta no contiene JSON: {content[:200]}")
    return json.loads(m.group(0))


# ══════════════════════════════════════════════════════════
# ESQUEMAS DE SALIDA (salida estructurada: JSON válido por construcción)
# ══════════════════════════════════════════════════════════

AgentName = Literal["GeoAgent", "AnalyticsAgent", "SpatialAgent", "RAGAgent", "QueryAgent", "MemoryAgent"]


class Tarea(BaseModel):
    agente: AgentName
    tarea: str = Field(description="instrucción concreta en español")

    # Un SLM suele escribir "spatial", "Geo Agent", "analytics"…: se normaliza ANTES de
    # validar el Literal. Sin esto, un nombre mal escrito invalidaba el plan completo.
    @field_validator("agente", mode="before")
    @classmethod
    def _agente(cls, v):
        return _normalize_agent(v)


class Plan(BaseModel):
    tareas: List[Tarea] = Field(min_length=1, max_length=4)


class Decision(BaseModel):
    decision: Literal["finalizar", "continuar"]
    respuesta_final: str = ""
    nueva_tarea: str = ""
    agente: AgentName = "GeoAgent"

    @field_validator("agente", mode="before")
    @classmethod
    def _agente(cls, v):
        return _normalize_agent(v) if v else "GeoAgent"


# ══════════════════════════════════════════════════════════
# NODO 1: PLANNER
# ══════════════════════════════════════════════════════════

PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Eres el Planner de un sistema multi-agente geoespacial sobre ArcGIS Online.
Analiza la solicitud y genera la lista MÍNIMA de tareas para resolverla.

AGENTES DISPONIBLES:
- GeoAgent       → ArcGIS Online en vivo: buscar capas/items, ver campos, consultar features con filtro SQL. Devuelve URLs de capas.
- AnalyticsAgent → estadísticas dentro de UNA capa (min, max, promedio, top N, conteo por categoría). Necesita la URL de la capa.
- SpatialAgent   → relaciones ESPACIALES entre DOS capas o respecto a un lugar: cuántos puntos caen en cada
                   polígono (accidentes por parroquia, denuncias en Quitumbe), qué polígonos cruza otra capa
                   (parroquias en zona de amenaza, provincias que atraviesa un parque), cercanía
                   (qué hay a menos de N km de un lugar, puntos cerca de otra capa), en qué zona está un lugar.
- RAGAgent       → preguntas sobre el contenido de WebMaps ya vectorizados (qué capas hay, qué mapa sirve para algo).
- QueryAgent     → consultas espaciales sobre resultados ya guardados en PostGIS local (buffer, más cercano).
- MemoryAgent    → historial de consultas anteriores del usuario.

REGLAS:
1. Entre 1 y 4 tareas. Una sola si basta.
2. Si hay un MAPA ABIERTO:
   - Preguntas sobre qué contiene el mapa, de qué trata o qué capas/campos tiene → RAGAgent.
   - Preguntas que piden DATOS de UNA capa (contar, filtrar, máximo, promedio, ranking,
     distribución) → AnalyticsAgent o GeoAgent.
   - Preguntas que cruzan DOS capas ("por parroquia", "en zona de", "dentro de", "atraviesa",
     "cerca de", "a menos de") → SpatialAgent, en UNA sola tarea con las dos capas.
   - NO busques las capas: escribe en la tarea el título y la URL de cada capa del mapa
     abierto que corresponda.
3. Si NO hay mapa abierto, NO agregues tareas para buscar capas: cada agente recibe
   automáticamente las capas candidatas del índice (URL, campos y valores) y, si le falta
   alguna, la busca él mismo.
4. Cada tarea es una instrucción concreta en español.
5. Si la pregunta nombra un TIPO de registro (denuncias, de seguridad, con fallecidos…):
   - con mapa abierto, escribe en la tarea el filtro exacto con un campo y un valor que APAREZCAN
     en el mapa abierto (campos de texto: campo = 'valor'; campos numéricos: campo > 0);
   - sin mapa abierto, describe el tipo en palabras ("solo los reportes de seguridad"): el agente
     ve los campos y valores reales y arma el filtro.
   Los campos de los EJEMPLOS son ficticios: nunca los copies.
6. No escribas coordenadas en la tarea: si la pregunta nombra un lugar, el SpatialAgent lo geocodifica.

EJEMPLOS (capas hipotéticas: "Incidentes" de puntos con campo tipo = robo|incendio,
"Barrios" y "Zonas inundables" de polígonos, "Hospitales" de puntos):
- "¿Qué barrio tiene más robos?" → UNA tarea SpatialAgent: contar los puntos de Incidentes (url)
  con filtro tipo = 'robo' dentro de cada polígono de Barrios (url).
- "¿Qué barrio tiene la mayor parte de su territorio en zona inundable?" → UNA tarea SpatialAgent:
  superponer Zonas inundables (url) sobre Barrios (url) y ordenar por porcentaje cubierto.
- "¿Cuántos incendios hay a menos de 300 m de un hospital?" → UNA tarea SpatialAgent: contar los
  puntos de Incidentes (url) con filtro tipo = 'incendio' a menos de 300 m de Hospitales (url).
- "¿Qué barrio tiene más incidentes con heridos?" (heridos es numérico) → UNA tarea SpatialAgent:
  contar los puntos de Incidentes (url) con filtro heridos > 0 dentro de cada polígono de Barrios (url).
- "¿Cuántos incidentes hay de cada tipo?" → UNA tarea AnalyticsAgent: conteo por el campo tipo de Incidentes (url).
- "¿Qué capas tiene este mapa?" → UNA tarea RAGAgent.
- "¿Qué hospital está más cerca de Cuenca?" → UNA tarea SpatialAgent: geocodificar Cuenca y buscar
  el elemento más cercano de Hospitales (url).

Responde SOLO JSON:
{{"tareas": [{{"agente": "GeoAgent", "tarea": "..."}}]}}"""),
    ("human", "MAPA ABIERTO:\n{mapa_abierto}\n\nSolicitud del usuario: {user_input}"),
])


def _mapa_abierto(webmap_id: Optional[str]) -> str:
    if not webmap_id:
        return "ninguno"
    from tools.vector_tools import webmap_layers_brief
    return webmap_layers_brief(webmap_id) or f"WebMap {webmap_id} (todavía no vectorizado)"


def node_planner(state: PlannerState) -> dict:
    logger.info("Planner: generando plan...")
    try:
        out = invoke_structured(PLAN_PROMPT, {"user_input": state["user_input"],
                                              "mapa_abierto": _mapa_abierto(state.get("webmap_id"))}, Plan)
        plan = [t.tarea for t in out.tareas]
        agentes = [_normalize_agent(t.agente) for t in out.tareas]
        for i, (a, t) in enumerate(zip(agentes, plan), 1):
            logger.info(f"  {i}. [{a}] {t}")
    except Exception as e:
        logger.error(f"Error en Planner, plan por defecto: {e}")
        # Con un mapa abierto, lo más seguro es describirlo; sin mapa, buscar en AGOL
        default = "RAGAgent" if state.get("webmap_id") else "GeoAgent"
        plan, agentes = [state["user_input"]], [default]

    return {"plan": plan, "agentes": agentes, "current_step": 0,
            "results": [], "steps": [], "sources": [], "should_end": False}


# ══════════════════════════════════════════════════════════
# NODO 2: EXECUTE
# ══════════════════════════════════════════════════════════

def _context_from(results: List[dict]) -> str:
    if not results:
        return ""
    bloques = [
        f"[Paso {r['step']} · {r['agente']}] {r['tarea']}\n{str(r['output'])[:CTX_POR_PASO]}"
        for r in results if r["status"] == "success"
    ]
    if not bloques:
        return ""
    return ("\n\nCONTEXTO DE PASOS ANTERIORES (úsalo, por ejemplo, para obtener "
            "URLs de capas o nombres de campos):\n" + "\n\n".join(bloques))


def node_execute(state: PlannerState) -> dict:
    step = state["current_step"]
    tarea = state["plan"][step]
    agente = state["agentes"][step] if step < len(state["agentes"]) else "GeoAgent"
    logger.info(f"Execute {step + 1}/{len(state['plan'])}: [{agente}] {tarea[:80]}")

    tarea_con_contexto = tarea + _context_from(state["results"])
    try:
        out = AGENTS[agente](tarea_con_contexto, state)
        status = "success" if out.get("success", True) else "error"
        output = out.get("output", "")
        tool_steps = out.get("steps", [])
        fuentes = out.get("sources", []) or []
    except Exception as e:
        logger.error(f"Error en paso {step + 1}: {e}")
        status, output, tool_steps, fuentes = "error", f"Error: {e}", [], []

    result = {"step": step + 1, "agente": agente, "tarea": tarea,
              "output": output, "status": status}

    if state.get("session_db_id"):
        MemoryAgent().log_agent_action(
            session_db_id=state["session_db_id"],
            agent=AGENT_ROLES.get(agente, AgentRole.PLANNER),
            action=tarea[:128],
            input_data={"tarea": tarea, "step": step + 1},
            output_data={"output": str(output)[:2000]},
            status=status,
        )

    tagged = [{**s, "agente": agente} for s in tool_steps]
    return {"results": state["results"] + [result],
            "steps": state.get("steps", []) + tagged,
            "sources": state.get("sources", []) + fuentes,
            "current_step": step + 1}


# ══════════════════════════════════════════════════════════
# NODO 3: REPLAN
# ══════════════════════════════════════════════════════════

REPLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Eres el evaluador de un sistema multi-agente geoespacial.
Con los resultados obtenidos decide si ya se puede responder al usuario.

Si alcanza:
{{"decision": "finalizar", "respuesta_final": "respuesta clara en español, con los datos concretos"}}

Si falta UNA tarea concreta:
{{"decision": "continuar", "nueva_tarea": "...", "agente": "GeoAgent|AnalyticsAgent|SpatialAgent|RAGAgent|QueryAgent|MemoryAgent"}}

No inventes datos que no estén en los resultados. Responde SOLO JSON, sin markdown."""),
    ("human", """Solicitud original: {user_input}

Resultados obtenidos:
{resultados}"""),
])


def node_replan(state: PlannerState) -> dict:
    logger.info("Replan: evaluando resultados...")
    results = state["results"]

    # Atajo para SLM: plan de una sola tarea que salió bien → la respuesta del
    # agente ya es la respuesta final. Se ahorra una llamada al modelo.
    if len(state["plan"]) == 1 and len(results) == 1 and results[0]["status"] == "success":
        return {"final_answer": results[0]["output"], "should_end": True}

    if results and all(r["status"] == "error" for r in results):
        errores = "; ".join(str(r["output"])[:200] for r in results)
        return {"final_answer": f"No se pudo completar la tarea. Errores: {errores}",
                "should_end": True}

    resultados = "\n\n".join(
        f"Paso {r['step']} ({r['agente']}, {r['status']}): {r['tarea']}\n{str(r['output'])[:CTX_POR_PASO]}"
        for r in results
    )
    try:
        d = invoke_structured(REPLAN_PROMPT, {"user_input": state["user_input"],
                                              "resultados": resultados}, Decision)

        if d.decision == "continuar" and len(state["plan"]) < MAX_TAREAS and d.nueva_tarea:
            agente = _normalize_agent(d.agente)
            logger.info(f"  → Replan agrega: [{agente}] {d.nueva_tarea[:80]}")
            return {"plan": state["plan"] + [d.nueva_tarea],
                    "agentes": state["agentes"] + [agente],
                    "should_end": False}

        final = d.respuesta_final or results[-1]["output"]
        return {"final_answer": final, "should_end": True}

    except Exception as e:
        logger.error(f"Error en Replan: {e}")
        outputs = "\n\n".join(str(r["output"]) for r in results if r["status"] == "success")
        return {"final_answer": outputs or "Tarea completada sin respuesta detallada.",
                "should_end": True}


# ══════════════════════════════════════════════════════════
# GRAFO
# ══════════════════════════════════════════════════════════

def _after_execute(state: PlannerState) -> str:
    return "execute" if state["current_step"] < len(state["plan"]) else "replan"


def _after_replan(state: PlannerState) -> str:
    return END if state.get("should_end") else "execute"


def build_planner_graph():
    graph = StateGraph(PlannerState)
    graph.add_node("planner", node_planner)
    graph.add_node("execute", node_execute)
    graph.add_node("replan", node_replan)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "execute")
    graph.add_conditional_edges("execute", _after_execute,
                                {"execute": "execute", "replan": "replan"})
    graph.add_conditional_edges("replan", _after_replan,
                                {"execute": "execute", END: END})
    return graph.compile()


_GRAPH = None


def run_planner(user_input: str, session_db_id: int = None,
                webmap_id: Optional[str] = None) -> dict:
    """
    Ejecuta Plan & Execute.

    Returns: dict con success, final_answer, plan, agentes, results,
             steps (pasos de herramientas, sirven para extraer GeoJSON) y steps_count.
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_planner_graph()

    initial: PlannerState = {
        "user_input": user_input, "webmap_id": webmap_id,
        "plan": [], "agentes": [], "current_step": 0,
        "results": [], "steps": [], "sources": [], "final_answer": "",
        "should_end": False, "session_db_id": session_db_id or 0,
    }

    try:
        final = _GRAPH.invoke(initial, {"recursion_limit": 30})
        if session_db_id:
            MemoryAgent().log_agent_action(
                session_db_id=session_db_id, agent=AgentRole.PLANNER,
                action="plan_and_execute",
                input_data={"user_input": user_input},
                output_data={"plan": final.get("plan", []),
                             "agentes": final.get("agentes", []),
                             "answer": final.get("final_answer", "")[:1000]},
                status="success",
            )
        return {
            "success"     : True,
            "final_answer": final.get("final_answer") or "Sin respuesta",
            "plan"        : final.get("plan", []),
            "agentes"     : final.get("agentes", []),
            "results"     : final.get("results", []),
            "steps"       : final.get("steps", []),
            "sources"     : final.get("sources", []),
            "steps_count" : len(final.get("results", [])),
        }
    except Exception as e:
        logger.error(f"Error en Planner graph: {e}")
        return {"success": False, "final_answer": f"Error en el Planner: {e}",
                "plan": [], "agentes": [], "results": [], "steps": [], "steps_count": 0}
