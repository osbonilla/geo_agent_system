"""
agents/base.py
──────────────
Fábrica común de agentes con tool calling nativo + grounding.

Por qué tool calling y no ReAct en texto
────────────────────────────────────────
ReAct en texto obliga al modelo a escribir exactamente
"Action: x / Action Input: y", y luego hay que parsearlo. Un SLM de 3-8B
rompe ese formato con frecuencia. Con tool calling el proveedor (Ollama,
NIM, OpenAI) devuelve la llamada como estructura: no hay nada que parsear.
Requiere un modelo con soporte de herramientas (en Ollama, los que tienen
la etiqueta "tools": qwen2.5, qwen3, llama3.1/3.2, mistral-nemo, etc.).

Grounding
─────────
Antes de llamar al modelo se recuperan del índice vectorial las capas más
parecidas a la pregunta, con su URL y sus campos reales. El SLM elige de
una lista corta en lugar de adivinar URLs y nombres de campos.
"""
import logging
import re
from typing import List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_settings
from llm.router import get_llm

logger = logging.getLogger(__name__)
settings = get_settings()


def grounding_context(question: str, webmap_id: Optional[str] = None) -> str:
    """
    Capas candidatas desde ChromaDB. Si hay un WebMap abierto, busca dentro de él;
    si no, en el índice global de capas. Devuelve "" si no hay nada útil.
    """
    try:
        from tools.vector_tools import search_webmap, compact_fields
        hits = search_webmap(question, webmap_id=webmap_id or "__layers__",
                             k=settings.grounding_k)
    except Exception as e:
        logger.debug(f"Grounding no disponible: {e}")
        return ""

    hits = [h for h in hits if h.get("metadata", {}).get("tipo") == "capa"
            and h.get("similitud", 0) >= settings.grounding_min_sim]
    if not hits:
        return ""

    bloques = []
    for h in hits:
        m = h["metadata"]
        texto = h["texto"]
        campos = texto.split("Campos:", 1)[1].strip() if "Campos:" in texto else ""
        campos = compact_fields(campos, 700)
        m_val = re.search(r"^Valores: (.*)$", texto, re.MULTILINE)
        bloques.append(
            f"- {m.get('capa_titulo')} (WebMap: {m.get('webmap_titulo')})\n"
            f"  url: {m.get('capa_url')}\n"
            f"  registros: {m.get('total_features') or 'n/d'} · geometría: {m.get('geometria') or 'n/d'}\n"
            f"  campos: {campos}"
            + (f"\n  valores: {m_val.group(1)}" if m_val else "")
        )
    return ("CAPAS CANDIDATAS (del índice de la organización; usa estas URLs y "
            "nombres de campo tal cual):\n" + "\n".join(bloques))


def build_agent(system_prompt: str, tools: List, max_iterations: Optional[int] = None) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm=get_llm(), tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=max_iterations or settings.agent_max_iterations,
        return_intermediate_steps=True,
    )


def _texto(output) -> str:
    """Algunos proveedores devuelven la salida como lista de bloques."""
    if isinstance(output, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in output)
    return str(output or "")


def _parece_llamada(texto: str) -> bool:
    """True si la "respuesta" es en realidad un JSON de argumentos de herramienta."""
    t = texto.strip().strip("`").removeprefix("json").strip()
    if not (t.startswith("{") and t.endswith("}")):
        return False
    try:
        import json
        d = json.loads(t)
    except ValueError:
        return False
    return isinstance(d, dict) and any(k.endswith("_url") or k in ("place", "lon", "lat") for k in d)


def _redactar(task: str, steps: List[dict]) -> str:
    """Respuesta final a partir de la salida de la última herramienta (una llamada al LLM)."""
    # El último resultado ÚTIL: no un error ni una consulta auxiliar (campos, búsqueda, geocodificación)
    auxiliares = {"get_layer_info", "search_layers", "geocode_place"}
    utiles = [s for s in steps if s["tool"] not in auxiliares and '"status": "error"' not in s["output"]]
    ultimo = (utiles or steps)[-1]
    prompt = (f"Pregunta: {task.split(chr(10) + chr(10))[0]}\n\n"
              f"Resultado de la herramienta {ultimo['tool']}:\n{ultimo['output'][:3000]}\n\n"
              "Responde la pregunta en español, breve, con los nombres y números exactos del resultado.")
    try:
        return _texto(get_llm().invoke(prompt).content)
    except Exception as e:
        logger.error(f"No se pudo redactar la respuesta: {e}")
        return ""


def run_agent(name: str, executor_factory, task: str, webmap_id: Optional[str] = None,
              grounding: bool = True, grounding_query: Optional[str] = None) -> dict:
    """
    Ejecuta un agente y normaliza la salida: success, output, steps, steps_count.
    grounding_query: texto para buscar capas candidatas (por defecto, la tarea).
    El Planner pasa la pregunta original + la subtarea: la subtarea sola suele ser
    demasiado corta ("calcular la altura máxima") para encontrar la capa correcta.
    """
    entrada = task
    ctx = grounding_context(grounding_query or task, webmap_id) if grounding else ""
    if ctx:
        entrada = f"{task}\n\n{ctx}"
    try:
        result = executor_factory().invoke({"input": entrada})
        # Algunos proveedores (gpt-oss en NIM) devuelven a veces una respuesta vacía sin llamar
        # a ninguna herramienta: se reintenta UNA vez. Las llamadas extra quedan en las métricas.
        if not result.get("intermediate_steps") and not _texto(result.get("output")).strip():
            logger.warning(f"{name}: respuesta vacía sin herramientas, se reintenta una vez")
            result = executor_factory().invoke({"input": entrada})
        # gpt-oss en NIM a veces escribe los argumentos de la herramienta como texto en vez
        # de llamarla ({"target_url": …}): se reintenta UNA vez.
        if _parece_llamada(_texto(result.get("output"))):
            logger.warning(f"{name}: escribió la llamada como texto, se reintenta una vez")
            previos = result.get("intermediate_steps", [])
            result = executor_factory().invoke({"input": entrada})
            result["intermediate_steps"] = previos + result.get("intermediate_steps", [])
        steps = [{"tool": a.tool, "input": a.tool_input, "output": str(obs)}
                 for a, obs in result.get("intermediate_steps", [])]
        output = _texto(result.get("output"))
        # Herramientas que funcionaron pero respuesta final vacía: se redacta con una llamada
        if not output.strip() and steps:
            output = _redactar(task, steps)
        output = output or "Sin respuesta"
        return {"success": True, "output": output, "steps": steps,
                "steps_count": len(steps), "grounding": bool(ctx)}
    except Exception as e:
        logger.error(f"Error en {name}: {e}")
        return {"success": False, "output": f"Error en {name}: {e}", "steps": [],
                "steps_count": 0, "grounding": bool(ctx)}
