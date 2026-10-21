"""
tests/test_planner.py
──────────────────────
Tests del Planner con LangGraph.

Uso:
  python tests/test_planner.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tools.utils import setup_logging
setup_logging("INFO")


def test_planner_simple():
    """Test básico — el Planner genera un plan y lo ejecuta."""
    print("\n" + "="*55)
    print("TEST 1: Planner — consulta simple")
    print("="*55)
    try:
        from agents.planner import run_planner
        result = run_planner("¿Cuántos features geoespaciales hay guardados en la base de datos?")

        print(f"  ✅ Plan generado: {len(result['plan'])} tareas")
        for i, tarea in enumerate(result["plan"], 1):
            print(f"     {i}. {tarea[:70]}")

        print(f"\n  ⚡ Pasos ejecutados: {result['steps_count']}")
        for r in result["results"]:
            icon = "✅" if r["status"] == "success" else "❌"
            print(f"     {icon} [{r['agente']}] {r['tarea'][:60]}")

        print(f"\n  📍 Respuesta final:")
        print(f"     {result['final_answer'][:200]}")
        return result["success"]
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_planner_multi_agente():
    """Test multi-agente — el Planner debe usar más de un agente."""
    print("\n" + "="*55)
    print("TEST 2: Planner — tarea multi-agente")
    print("="*55)
    try:
        from agents.planner import run_planner
        result = run_planner(
            "Busca capas de arbolado en ArcGIS Online y luego "
            "dime cuántos features tenemos guardados localmente"
        )

        agentes_usados = list(set(r["agente"] for r in result["results"]))
        print(f"  ✅ Agentes coordinados: {agentes_usados}")
        print(f"  ✅ Pasos ejecutados: {result['steps_count']}")
        print(f"\n  📍 Respuesta:")
        print(f"     {result['final_answer'][:300]}")

        return result["success"]
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n🧪 TESTS DEL PLANNER (LangGraph)")
    print("=" * 55)
    print("⚠️  Estos tests consumen tokens de Groq\n")

    results = {
        "planner_simple"      : test_planner_simple(),
        "planner_multi_agente": test_planner_multi_agente(),
    }

    print("\n" + "="*55)
    print("RESUMEN")
    print("="*55)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  {passed}/{len(results)} tests pasaron\n")
