"""
main.py
────────
CLI interactivo del sistema multi-agente geoespacial.

Comandos:
  plan: <tarea>     → Planner completo (Plan & Execute con LangGraph)
  geo: <tarea>      → GeoAgent directo (ArcGIS Online)
  query: <tarea>    → QueryAgent directo (PostGIS local)
  rag: <pregunta>   → RAGAgent sobre WebMaps vectorizados
  vec: mapas        → Listar WebMaps de ArcGIS Online
  vec: <item_id>    → Vectorizar un WebMap
  vec: todos        → Vectorizar todos los WebMaps
  vec: estado       → Estado del índice vectorial
  llm               → Ver el modelo activo
  mem: historial    → Ver sesiones recientes
  mem: buscar <kw>  → Buscar en historial
  mem: check        → Verificar base de datos
  test              → Correr tests de conexión
  ayuda             → Ver comandos
  salir             → Terminar
"""
import os
import sys
import json
import time
import logging
from dotenv import load_dotenv

load_dotenv()

from tools.utils import setup_logging
setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def print_banner():
    print("\n" + "="*58)
    print("  SISTEMA MULTI-AGENTE GEOESPACIAL")
    print("  LangGraph · ArcGIS Online · ChromaDB · LLM intercambiable")
    print("="*58)


def print_help():
    print("\n  Comandos disponibles:")
    print("  ──────────────────────────────────────────────────")
    print("  plan: <tarea>     → Planner completo (Plan & Execute)")
    print("  geo: <tarea>      → GeoAgent — ArcGIS Online")
    print("  query: <tarea>    → QueryAgent — PostGIS local")
    print("  rag: <pregunta>   → RAGAgent — WebMaps vectorizados")
    print("  vec: mapas        → Listar WebMaps de ArcGIS Online")
    print("  vec: <item_id>    → Vectorizar ese WebMap en ChromaDB")
    print("  vec: todos        → Vectorizar todos los WebMaps")
    print("  vec: estado       → Estado del índice vectorial")
    print("  llm               → Modelo activo (proveedor/modelo)")
    print("  mem: historial    → Ver últimas 5 sesiones")
    print("  mem: buscar <kw>  → Buscar sesiones por keyword")
    print("  mem: check        → Verificar base de datos")
    print("  test              → Verificar todas las conexiones")
    print("  ayuda             → Mostrar esta ayuda")
    print("  salir             → Terminar el sistema")
    print()
    print("  Ejemplos Planner (recomendado):")
    print("  plan: busca la capa de arbolado y dime cuántos registros tiene")
    print("  plan: ¿qué capas tengo en ArcGIS y cuántos features guardé?")
    print("  plan: analiza la capa de encuesta arbolado de Quito")
    print()
    print("  Ejemplos directos:")
    print("  geo: ¿qué capas tengo disponibles?")
    print("  query: ¿cuántos features hay guardados por capa?")
    print("  ──────────────────────────────────────────────────\n")


def handle_planner(task: str):
    """
    Ejecuta el Planner completo — Plan & Execute con LangGraph.
    Orquesta GeoAgent, QueryAgent y MemoryAgent automáticamente.
    """
    from agents.planner import run_planner
    from agents.memory_agent import MemoryAgent
    from db.models import AgentRole

    mem = MemoryAgent()
    session_info = mem.save_session(f"[PLANNER] {task}")
    session_db_id = session_info.get("db_id")

    print(f"\n Planner iniciando Plan & Execute...")
    print(f"  Solicitud: {task}\n")
    print("  " + "="*50)

    start = time.time()
    result = run_planner(task, session_db_id=session_db_id)
    duration_ms = int((time.time() - start) * 1000)

    print("  " + "="*50)

    # Mostrar plan generado
    if result["plan"]:
        print(f"\n Plan generado ({len(result['plan'])} tareas):")
        for i, tarea in enumerate(result["plan"], 1):
            print(f"     {i}. {tarea[:80]}")

    # Mostrar ejecución paso a paso
    if result["results"]:
        print(f"\n Ejecución ({result['steps_count']} pasos):")
        for r in result["results"]:
            icon = "Ok" if r["status"] == "success" else "Error"
            print(f"     {icon} Paso {r['step']} [{r['agente']}]: {r['tarea'][:60]}")

    # Mostrar respuesta final
    print(f"\n Respuesta final:\n")
    print(f"  {result['final_answer']}\n")

    print(f" Tiempo total: {duration_ms/1000:.1f}s")
    if session_db_id:
        print(f" Guardado en DB — sesión #{session_db_id}\n")

    if not result["success"]:
        print(" El Planner encontró un error.\n")


def handle_geo_agent(task: str):
    from agents.geo_agent import run_geo_agent
    from agents.memory_agent import MemoryAgent
    from db.models import AgentRole

    mem = MemoryAgent()
    session_info = mem.save_session(task)
    session_db_id = session_info.get("db_id")

    print(f"\n GeoAgent procesando...")
    print(f"  Tarea: {task}\n")
    print("  " + "-"*50)

    start = time.time()
    result = run_geo_agent(task)
    duration_ms = int((time.time() - start) * 1000)

    print("  " + "-"*50)

    if session_db_id:
        mem.log_agent_action(
            session_db_id=session_db_id,
            agent=AgentRole.GEO_AGENT,
            action="run_geo_agent",
            input_data={"task": task},
            output_data={"output": result["output"][:500]},
            status="success" if result["success"] else "error",
            duration_ms=duration_ms,
        )

    if result["steps"]:
        print(f"\n  🔧 Tools ({result['steps_count']} pasos):")
        for i, step in enumerate(result["steps"], 1):
            print(f"     {i}. {step['tool']}({str(step['input'])[:60]})")

    print(f"\n Respuesta:\n  {result['output']}\n")
    if session_db_id:
        print(f" Guardado en DB — sesión #{session_db_id}\n")


def handle_query_agent(task: str):
    from agents.query_agent import run_query_agent
    from agents.memory_agent import MemoryAgent
    from db.models import AgentRole

    mem = MemoryAgent()
    session_info = mem.save_session(f"[QUERY] {task}")
    session_db_id = session_info.get("db_id")

    print(f"\n QueryAgent procesando...")
    print(f"  Tarea: {task}\n")
    print("  " + "-"*50)

    start = time.time()
    result = run_query_agent(task)
    duration_ms = int((time.time() - start) * 1000)

    print("  " + "-"*50)

    if session_db_id:
        mem.log_agent_action(
            session_db_id=session_db_id,
            agent=AgentRole.QUERY_AGENT,
            action="run_query_agent",
            input_data={"task": task},
            output_data={"output": result["output"][:500]},
            status="success" if result["success"] else "error",
            duration_ms=duration_ms,
        )

    if result["steps"]:
        print(f"\n Tools ({result['steps_count']} pasos):")
        for i, step in enumerate(result["steps"], 1):
            print(f"     {i}. {step['tool']}({str(step['input'])[:60]})")

    print(f"\n Respuesta:\n  {result['output']}\n")
    if session_db_id:
        print(f" Guardado en DB — sesión #{session_db_id}\n")


def handle_memory_historial():
    from agents.memory_agent import MemoryAgent
    mem = MemoryAgent()
    sessions = mem.get_recent_sessions(limit=5)
    if not sessions:
        print(" No hay sesiones guardadas aún.")
        return
    print(f"\n Últimas {len(sessions)} sesiones:\n")
    for s in sessions:
        print(f"  [{s['db_id']}] {s['created_at'][:19]}")
        print(f"       → {s['user_input'][:80]}")
    print()


def handle_memory_buscar(keyword: str):
    from agents.memory_agent import run_memory_agent
    if not keyword:
        print(" Escribe una palabra clave. Ejemplo: mem: buscar arbolado")
        return
    print(f"\n Buscando sesiones con '{keyword}'...\n")
    result = run_memory_agent(f"Busca sesiones anteriores que contengan la palabra: {keyword}")
    print(f"  {result['output']}\n")


def handle_memory_check():
    from agents.memory_agent import MemoryAgent
    mem = MemoryAgent()
    status = mem.check_db()
    if status["status"] == "ok":
        print(f" PostgreSQL OK | PostGIS: {status['postgis']}")
    else:
        print(f" DB Error: {status.get('message', 'desconocido')}")



def handle_llm_info():
    from llm.router import get_llm_info, available_providers
    info = get_llm_info()
    if info.get("ok"):
        print(f"\n  Modelo activo : {info['full']}")
        print(f"  Proveedor     : {info['label']}")
        print(f"  Local         : {'sí' if info['local'] else 'no'}")
        print(f"  Elegido por   : {info['source']}\n")
    else:
        print(f"\n  {info.get('error')}\n")
    print("  Proveedores:")
    for p in available_providers():
        marca = "listo" if p["ready"] else f"falta {p['env_key']}"
        print(f"    {p['name']:<12} {marca:<28} ej: LLM_MODEL={p['example']}")
    print()


def handle_vec(arg: str):
    from tools.vector_tools import (
        list_webmaps, vectorize_webmap, vectorize_all_webmaps, vector_status,
    )

    arg = (arg or "").strip()

    if arg in ("mapas", "list", "webmaps"):
        mapas = list_webmaps(max_items=25)
        if not mapas:
            print("  No hay WebMaps visibles para este usuario.")
            return
        print(f"\n  {len(mapas)} WebMaps en ArcGIS Online:\n")
        for m in mapas:
            print(f"    {m['id']}  {m['titulo'][:55]}")
        print("\n  Vectorizá uno con: vec: <item_id>\n")
        return

    if arg in ("estado", "status"):
        st = vector_status()
        if not st.get("ok"):
            print(f"  ChromaDB no responde: {st.get('error')}")
            return
        print(f"\n  ChromaDB  : {st['modo']} — {st['ubicacion']}")
        print(f"  Colecciones: {st['colecciones']}   Documentos: {st['documentos']}")
        for w in st.get("webmaps", []):
            print(f"    · {w.get('webmap_titulo')} — {w.get('total_capas')} capas "
                  f"[{w.get('embedding')}]")
        print()
        return

    if arg in ("todos", "all"):
        print("\n  Vectorizando todos los WebMaps...\n")
        r = vectorize_all_webmaps(max_items=20, deep=False)
        print(f"  Vectorizados: {r['vectorizados']}   Fallidos: {r['fallidos']}\n")
        for e in r["errores"]:
            print(f"    Error en {e['titulo']}: {e['error']}")
        return

    if not arg:
        print("  Usá: vec: mapas | vec: <item_id> | vec: todos | vec: estado")
        return

    print(f"\n  Vectorizando WebMap {arg}...\n")
    r = vectorize_webmap(arg, deep=True)
    print(f"  {r['titulo']}")
    print(f"  Capas       : {r['capas']}")
    print(f"  Documentos  : {r['documentos']}")
    print(f"  Colección   : {r['coleccion']}")
    print(f"  Embeddings  : {r['embedding']}\n")


def handle_rag(question: str, webmap_id: str = None):
    from agents.rag_agent import run_rag_agent
    print(f"\n RAGAgent buscando en el índice...\n")
    r = run_rag_agent(question, webmap_id=webmap_id)
    print(f"  {r['output']}\n")
    for s in r.get("sources", []):
        print(f"    capa citada: {s['capa']} ({s['registros']} registros)")
    print()


def main():
    from db.models import init_db
    db = init_db()
    if not db["ok"]:
        print(f"  Aviso: PostgreSQL no disponible ({db['error'][:120]})")
    print_banner()
    print_help()

    while True:
        try:
            user_input = input("Agente IA > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Hasta luego!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "salir":
            print("  Hasta luego!")
            break
        elif cmd == "ayuda":
            print_help()
        elif cmd == "test":
            os.system(f"{sys.executable} tests/test_arcgis_connection.py")
        elif cmd == "mem: check":
            handle_memory_check()
        elif cmd == "mem: historial":
            handle_memory_historial()
        elif cmd.startswith("mem: buscar"):
            keyword = user_input[11:].strip()
            handle_memory_buscar(keyword)
        elif cmd.startswith("plan:"):
            task = user_input[5:].strip()
            if not task:
                print(" Escribe una tarea. Ejemplo: plan: analiza la capa de arbolado")
            else:
                handle_planner(task)
        elif cmd.startswith("geo:"):
            task = user_input[4:].strip()
            if not task:
                print(" Escribe una tarea. Ejemplo: geo: ¿qué capas tengo?")
            else:
                handle_geo_agent(task)
        elif cmd.startswith("vec:"):
            handle_vec(user_input[4:].strip())
        elif cmd == "llm":
            handle_llm_info()
        elif cmd.startswith("rag:"):
            pregunta = user_input[4:].strip()
            if not pregunta:
                print(" Escribí una pregunta. Ejemplo: rag: ¿qué capas de riesgo tiene el mapa?")
            else:
                handle_rag(pregunta)
        elif cmd.startswith("query:"):
            task = user_input[6:].strip()
            if not task:
                print(" Escribe una tarea. Ejemplo: query: ¿cuántos features hay?")
            else:
                handle_query_agent(task)
        else:
            print(" Comando no reconocido. Escribe 'ayuda' para ver los comandos.")


if __name__ == "__main__":
    main()
