"""
tests/test_memory_agent.py
───────────────────────────
Tests del MemoryAgent — verifica persistencia en PostgreSQL + PostGIS.

Uso:
  python tests/test_memory_agent.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tools.utils import setup_logging
setup_logging("INFO")


def test_save_session():
    """Guarda una sesión y verifica que se creó en la DB."""
    print("\n" + "="*55)
    print("TEST 1: Guardar sesión")
    print("="*55)
    try:
        from agents.memory_agent import MemoryAgent
        mem = MemoryAgent()
        result = mem.save_session("Test: ¿qué capas tengo disponibles?")
        assert "db_id" in result
        assert "session_id" in result
        print(f"  ✅ Sesión guardada — db_id: {result['db_id']}")
        print(f"  ✅ session_id: {result['session_id']}")
        return result["db_id"]
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def test_log_action(session_db_id: int):
    """Registra un log de agente."""
    print("\n" + "="*55)
    print("TEST 2: Registrar log de agente")
    print("="*55)
    try:
        from agents.memory_agent import MemoryAgent
        from db.models import AgentRole
        mem = MemoryAgent()
        ok = mem.log_agent_action(
            session_db_id=session_db_id,
            agent=AgentRole.GEO_AGENT,
            action="list_content",
            input_data={"max_items": 10},
            output_data={"total": 5, "items": ["capa1", "capa2"]},
            status="success",
            duration_ms=1250,
        )
        assert ok
        print(f"  ✅ Log guardado para sesión {session_db_id}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_save_geo_result(session_db_id: int):
    """Persiste un resultado geoespacial con geometría."""
    print("\n" + "="*55)
    print("TEST 3: Guardar resultado geoespacial (PostGIS)")
    print("="*55)
    try:
        from agents.memory_agent import MemoryAgent
        mem = MemoryAgent()
        ok = mem.save_geo_result(
            session_db_id=session_db_id,
            layer_name="Arbolado Quito - Test",
            attributes={"especie": "Eucalipto", "altura": 12.5},
            geometry={"x": -78.4678, "y": -0.1807},  # Quito centro
            feature_id="test_001",
            source_url="https://services2.arcgis.com/test/FeatureServer/0",
        )
        assert ok
        print(f"  ✅ GeoResult guardado con geometría POINT en PostGIS")
        print(f"     Coordenadas: -78.4678, -0.1807 (Quito centro)")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_get_recent_sessions():
    """Recupera sesiones recientes."""
    print("\n" + "="*55)
    print("TEST 4: Recuperar sesiones recientes")
    print("="*55)
    try:
        from agents.memory_agent import MemoryAgent
        mem = MemoryAgent()
        sessions = mem.get_recent_sessions(limit=3)
        print(f"  ✅ Sesiones recuperadas: {len(sessions)}")
        for s in sessions:
            print(f"     [{s['db_id']}] {s['user_input'][:60]}")
        return len(sessions) > 0
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_get_geo_results(session_db_id: int):
    """Recupera resultados geoespaciales de la sesión."""
    print("\n" + "="*55)
    print("TEST 5: Recuperar resultados geoespaciales")
    print("="*55)
    try:
        from agents.memory_agent import MemoryAgent
        mem = MemoryAgent()
        results = mem.get_geo_results_by_session(session_db_id)
        print(f"  ✅ GeoResults de sesión {session_db_id}: {len(results)}")
        for r in results:
            print(f"     - {r['layer_name']} | feature: {r['feature_id']}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n🧪 TESTS DEL MEMORY AGENT")
    print("=" * 55)

    session_db_id = test_save_session()

    if session_db_id:
        results = {
            "save_session"    : True,
            "log_action"      : test_log_action(session_db_id),
            "save_geo_result" : test_save_geo_result(session_db_id),
            "recent_sessions" : test_get_recent_sessions(),
            "geo_results"     : test_get_geo_results(session_db_id),
        }
    else:
        results = {"save_session": False}

    print("\n" + "="*55)
    print("RESUMEN")
    print("="*55)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  {passed}/{len(results)} tests pasaron\n")
