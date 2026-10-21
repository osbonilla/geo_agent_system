"""
tests/test_geo_agent.py
────────────────────────
Tests del GeoAgent — verifica que el agente razona y usa tools correctamente.

Uso:
  python tests/test_geo_agent.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tools.utils import setup_logging
setup_logging("INFO")


def test_geo_agent_conexion():
    """Verifica que el agente puede verificar la conexión ArcGIS."""
    print("\n" + "="*55)
    print("TEST 1: GeoAgent — verificar conexión ArcGIS")
    print("="*55)
    try:
        from agents.geo_agent import run_geo_agent
        result = run_geo_agent("Verifica la conexión con ArcGIS Online y dime quién soy")
        print(f"  ✅ Respuesta obtenida")
        print(f"  📍 {result['output'][:200]}")
        print(f"  🔧 Tools usadas: {result['steps_count']}")
        return result["success"]
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_geo_agent_listar():
    """Verifica que el agente puede listar capas."""
    print("\n" + "="*55)
    print("TEST 2: GeoAgent — listar capas disponibles")
    print("="*55)
    try:
        from agents.geo_agent import run_geo_agent
        result = run_geo_agent("¿Qué capas o items tengo disponibles en ArcGIS Online?")
        print(f"  ✅ Respuesta obtenida")
        print(f"  📍 {result['output'][:300]}")
        print(f"  🔧 Tools usadas: {result['steps_count']}")
        return result["success"]
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_geo_agent_buscar():
    """Verifica que el agente puede buscar capas por keyword."""
    print("\n" + "="*55)
    print("TEST 3: GeoAgent — buscar capa de Arbolado")
    print("="*55)
    try:
        from agents.geo_agent import run_geo_agent
        result = run_geo_agent("Busca capas relacionadas con arbolado o árboles en ArcGIS")
        print(f"  ✅ Respuesta obtenida")
        print(f"  📍 {result['output'][:300]}")
        print(f"  🔧 Tools usadas: {result['steps_count']}")
        return result["success"]
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n🧪 TESTS DEL GEOAGENT")
    print("=" * 55)

    results = {
        "conexion" : test_geo_agent_conexion(),
        "listar"   : test_geo_agent_listar(),
        "buscar"   : test_geo_agent_buscar(),
    }

    print("\n" + "="*55)
    print("RESUMEN")
    print("="*55)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  {passed}/{len(results)} tests pasaron\n")
