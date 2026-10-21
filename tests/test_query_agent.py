"""
tests/test_query_agent.py
──────────────────────────
Tests del QueryAgent — verifica consultas espaciales PostGIS.

Uso:
  python tests/test_query_agent.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tools.utils import setup_logging
setup_logging("INFO")


def test_count_by_layer():
    """Cuenta features por capa directamente con la tool."""
    print("\n" + "="*55)
    print("TEST 1: Contar features por capa")
    print("="*55)
    try:
        from tools.postgis_tools import count_by_layer
        import json
        result = json.loads(count_by_layer.invoke("all"))
        print(f"  ✅ Capas en DB: {result['total_capas']}")
        for capa in result.get("capas", []):
            print(f"     - {capa['layer_name']}: {capa['total']} features")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_query_geo_results():
    """Consulta features guardados."""
    print("\n" + "="*55)
    print("TEST 2: Consultar geo_results")
    print("="*55)
    try:
        from tools.postgis_tools import query_geo_results
        import json
        result = json.loads(query_geo_results.invoke('{"limit": 5}'))
        print(f"  ✅ Features encontrados: {result['total']}")
        for f in result.get("features", []):
            print(f"     - [{f['id']}] {f['layer_name']} | {f['feature_id']}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_spatial_buffer():
    """Busca features en radio de 50km de Quito."""
    print("\n" + "="*55)
    print("TEST 3: Buffer espacial desde Quito centro")
    print("="*55)
    try:
        from tools.postgis_tools import spatial_buffer
        import json
        result = json.loads(spatial_buffer.invoke(
            '{"lon": -78.4678, "lat": -0.1807, "radio_metros": 50000}'
        ))
        print(f"  ✅ Features en radio 50km: {result['total_encontrado']}")
        for f in result.get("features", []):
            print(f"     - {f['layer_name']} a {f['distancia_metros']}m")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_get_stats():
    """Estadísticas de la sesión 1."""
    print("\n" + "="*55)
    print("TEST 4: Estadísticas de sesión")
    print("="*55)
    try:
        from tools.postgis_tools import get_stats_by_session
        import json
        result = json.loads(get_stats_by_session.invoke("1"))
        print(f"  ✅ Stats sesión 1:")
        print(f"     Agentes: {len(result.get('agentes', []))}")
        geo = result.get("geo_results", {})
        print(f"     Features: {geo.get('total_features', 0)}")
        print(f"     Con geometría: {geo.get('con_geometria', 0)}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_query_agent_llm():
    """Prueba el QueryAgent con LLM."""
    print("\n" + "="*55)
    print("TEST 5: QueryAgent con LLM")
    print("="*55)
    try:
        from agents.query_agent import run_query_agent
        result = run_query_agent("¿Cuántos features hay guardados por capa en la base de datos?")
        print(f"  ✅ Respuesta obtenida")
        print(f"  📍 {result['output'][:200]}")
        print(f"  🔧 Tools usadas: {result['steps_count']}")
        return result["success"]
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n🧪 TESTS DEL QUERY AGENT")
    print("=" * 55)

    results = {
        "count_by_layer"   : test_count_by_layer(),
        "query_geo_results": test_query_geo_results(),
        "spatial_buffer"   : test_spatial_buffer(),
        "get_stats"        : test_get_stats(),
        "query_agent_llm"  : test_query_agent_llm(),
    }

    print("\n" + "="*55)
    print("RESUMEN")
    print("="*55)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  {passed}/{len(results)} tests pasaron\n")
