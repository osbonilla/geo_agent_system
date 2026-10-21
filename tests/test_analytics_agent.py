"""
tests/test_analytics_agent.py
──────────────────────────────
Tests del AnalyticsAgent.

Uso:
  python tests/test_analytics_agent.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from tools.utils import setup_logging
setup_logging("INFO")

LAYER_URL = "https://services2.arcgis.com/kHvSRXWQHCjXtMkT/arcgis/rest/services/survey123_229156bed8894099980291845f5aea37_results/FeatureServer/0"


def test_layer_summary():
    print("\n" + "="*55)
    print("TEST 1: Resumen de la capa")
    print("="*55)
    try:
        from tools.analytics_tools import layer_summary
        import json
        result = json.loads(layer_summary.invoke({"layer_url": LAYER_URL}))
        print(f"  ✅ Total features    : {result.get('total_registros')}")
        print(f"  ✅ Tipo geometría    : {result.get('geometria')}")
        print(f"  ✅ Campos numéricos  : {len(result.get('campos_numericos', []))}")
        print(f"  ✅ Campos categóricos: {len(result.get('campos_texto', []))}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_field_statistics():
    print("\n" + "="*55)
    print("TEST 2: Estadísticas de altura_total")
    print("="*55)
    try:
        from tools.analytics_tools import field_statistics
        import json
        result = json.loads(field_statistics.invoke(
            {"layer_url": LAYER_URL, "field_name": "_3_altura_total"}
        ))
        if result.get("status") == "sin_datos":
            print(f"  ⚠️  Capa vacía — sin features")
            return True
        print(f"  ✅ Mínimo  : {result.get('minimo')}")
        print(f"  ✅ Máximo  : {result.get('maximo')}")
        print(f"  ✅ Promedio: {result.get('promedio')}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_count_by_category():
    print("\n" + "="*55)
    print("TEST 3: Conteo por especie")
    print("="*55)
    try:
        from tools.analytics_tools import count_by_category
        import json
        result = json.loads(count_by_category.invoke(
            {"layer_url": LAYER_URL, "field_name": "field_7"}
        ))
        print(f"  ✅ Categorías: {result.get('total_categorias')}")
        for c in result.get("categorias", [])[:3]:
            print(f"     - {c['categoria']}: {c['conteo']}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_analytics_agent_llm():
    print("\n" + "="*55)
    print("TEST 4: AnalyticsAgent con LLM")
    print("="*55)
    try:
        from agents.analytics_agent import run_analytics_agent
        result = run_analytics_agent(
            f"Genera un resumen de la capa de arbolado. URL: {LAYER_URL}"
        )
        print(f"  ✅ Respuesta obtenida")
        print(f"  📍 {result['output'][:200]}")
        print(f"  🔧 Tools usadas: {result['steps_count']}")
        return result["success"]
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n🧪 TESTS DEL ANALYTICS AGENT")
    print("=" * 55)
    results = {
        "layer_summary"      : test_layer_summary(),
        "field_statistics"   : test_field_statistics(),
        "count_by_category"  : test_count_by_category(),
        "analytics_agent_llm": test_analytics_agent_llm(),
    }
    print("\n" + "="*55)
    print("RESUMEN")
    print("="*55)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  {passed}/{len(results)} tests pasaron\n")
