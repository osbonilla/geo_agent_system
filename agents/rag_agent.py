"""
agents/rag_agent.py
────────────────────
RAGAgent: responde preguntas sobre un WebMap vectorizado.

Por qué no es un ReAct
──────────────────────
Recuperar y responder es un flujo de un solo paso. Meterlo en un bucle
Thought/Action/Observation con un SLM de 7B agrega errores de formato sin
mejorar la respuesta. Acá se recupera primero y se le pide al modelo una
sola cosa: redactar con el contexto recuperado.

Devuelve además las capas citadas, para que el frontend pueda agregarlas
al mapa sin que el usuario tenga que buscarlas.
"""
import json
import logging
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate

from llm.router import get_llm
from tools.vector_tools import search_webmap, read_webmap

logger = logging.getLogger(__name__)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Eres el RAGAgent de un sistema geoespacial. Respondés preguntas
sobre mapas web de ArcGIS Online usando únicamente el contexto recuperado del
índice vectorial.

Reglas:
1. Respondé en español, claro y concreto.
2. Usá solo los datos del contexto. Si el contexto no alcanza, decilo y sugerí
   qué capa o dato haría falta.
3. Cuando menciones una capa, escribí su título tal como aparece en el contexto.
4. Si hay números (registros, campos), citalos.
5. Nada de inventar URLs, conteos ni nombres de campos.
6. Máximo 6 frases, salvo que pidan un listado."""),
    ("human", """Pregunta: {question}

Contexto recuperado del WebMap{webmap_label}:
{context}

Respondé la pregunta."""),
])


def _format_context(hits: List[dict]) -> str:
    if not hits:
        return "(el índice no devolvió resultados)"
    bloques = []
    for i, h in enumerate(hits, 1):
        meta = h.get("metadata", {})
        bloques.append(
            f"[{i}] similitud {h['similitud']} · {meta.get('tipo', 'doc')}\n{h['texto']}"
        )
    return "\n\n".join(bloques)


def _sources_from_hits(hits: List[dict]) -> List[dict]:
    vistos, fuentes = set(), []
    for h in hits:
        meta = h.get("metadata", {})
        if meta.get("tipo") != "capa":
            continue
        url = meta.get("capa_url", "")
        if not url or url in vistos:
            continue
        vistos.add(url)
        fuentes.append({
            "capa"      : meta.get("capa_titulo", ""),
            "url"       : url,
            "webmap_id" : meta.get("webmap_id", ""),
            "geometria" : meta.get("geometria", ""),
            "registros" : meta.get("total_features", 0),
            "similitud" : h.get("similitud"),
        })
    return fuentes


def run_rag_agent(question: str, webmap_id: Optional[str] = None, k: int = 6) -> dict:
    """
    Responde una pregunta usando el índice vectorial.

    Args:
        question : pregunta en lenguaje natural
        webmap_id: si viene, busca solo dentro de ese WebMap;
                   si no, busca en el catálogo global de WebMaps
        k        : cuántos fragmentos recuperar
    """
    try:
        hits = search_webmap(question, webmap_id=webmap_id, k=k)
    except Exception as e:
        logger.error(f"Error consultando el índice vectorial: {e}")
        return {
            "success"    : False,
            "output"     : (
                f"No pude consultar el índice vectorial: {e}. "
                f"Verificá que ChromaDB esté levantado y que el WebMap esté vectorizado."
            ),
            "steps"      : [],
            "steps_count": 0,
            "sources"    : [],
        }

    if not hits:
        destino = f"el WebMap {webmap_id}" if webmap_id else "el catálogo"
        return {
            "success"    : True,
            "output"     : (
                f"No hay nada indexado para {destino}. Vectorizá el mapa primero "
                f"con el botón «Vectorizar» o con POST /api/ingest-webmap."
            ),
            "steps"      : [{"tool": "search_webmap_context", "input": question, "output": "0 resultados"}],
            "steps_count": 1,
            "sources"    : [],
        }

    contexto = _format_context(hits)
    etiqueta = ""
    if webmap_id:
        titulo = hits[0].get("metadata", {}).get("webmap_titulo", "")
        etiqueta = f" «{titulo}»" if titulo else ""

    try:
        llm = get_llm()
        chain = RAG_PROMPT | llm
        respuesta = chain.invoke({
            "question"     : question,
            "context"      : contexto,
            "webmap_label" : etiqueta,
        })
        texto = respuesta.content.strip() if hasattr(respuesta, "content") else str(respuesta)
    except Exception as e:
        logger.error(f"Error del LLM en RAGAgent: {e}")
        return {
            "success"    : False,
            "output"     : f"Error del LLM: {e}",
            "steps"      : [{"tool": "search_webmap_context", "input": question,
                             "output": f"{len(hits)} fragmentos recuperados"}],
            "steps_count": 1,
            "sources"    : _sources_from_hits(hits),
        }

    return {
        "success"    : True,
        "output"     : texto,
        "steps"      : [{
            "tool"  : "search_webmap_context",
            "input" : json.dumps({"query": question, "webmap_id": webmap_id, "k": k}, ensure_ascii=False),
            "output": f"{len(hits)} fragmentos · mejor similitud {hits[0]['similitud']}",
        }],
        "steps_count": 1,
        "sources"    : _sources_from_hits(hits),
    }


def run_rag_agent_simple(task: str) -> str:
    """Versión de una línea — la usa el Planner."""
    return run_rag_agent(task)["output"]


def describe_webmap(webmap_id: str) -> dict:
    """Resumen redactado de un WebMap, sin pasar por el índice."""
    wm = read_webmap(webmap_id)
    capas = "\n".join(
        f"- {c['titulo']} ({c['tipo'] or 'capa'})" for c in wm["capas"][:40]
    )
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Describí mapas web de ArcGIS en español, en 4 frases, para un analista GIS."),
        ("human", "Mapa: {titulo}\nResumen: {snippet}\nCapas:\n{capas}"),
    ])
    resp = (prompt | llm).invoke({
        "titulo" : wm["titulo"],
        "snippet": wm["snippet"],
        "capas"  : capas,
    })
    return {
        "webmap_id" : wm["id"],
        "titulo"    : wm["titulo"],
        "total_capas": wm["total_capas"],
        "descripcion": resp.content if hasattr(resp, "content") else str(resp),
    }
