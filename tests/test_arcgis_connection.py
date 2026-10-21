"""
tests/test_arcgis_connection.py
────────────────────────────────
Script de verificación de conexiones.
Corre esto para confirmar que todo funciona antes de avanzar.

Uso:
  cd geo_agent_system
  python tests/test_arcgis_connection.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tools.utils import setup_logging
setup_logging("INFO")

import logging
logger = logging.getLogger(__name__)


def test_settings():
    """Verifica que las variables de entorno están cargadas."""
    print("\n" + "="*50)
    print("TEST 1: Variables de entorno")
    print("="*50)
    try:
        from config.settings import get_settings
        s = get_settings()
        print(f"  ✅ ARCGIS_USERNAME  : {s.arcgis_username}")
        print(f"  ✅ ARCGIS_URL       : {s.arcgis_url}")
        print(f"  ✅ DB_HOST          : {s.db_host}:{s.db_port}/{s.db_name}")
        print(f"  ✅ LLM_MODEL        : {s.llm_model}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        print("     → Revisa que tu archivo .env esté configurado")
        return False


def test_arcgis_connection():
    """Verifica la conexión a ArcGIS Online."""
    print("\n" + "="*50)
    print("TEST 2: Conexión ArcGIS Online")
    print("="*50)
    try:
        from tools.arcgis_tools import get_gis_connection
        gis = get_gis_connection()
        user = gis.properties.user
        print(f"  ✅ Conectado como   : {user.username}")
        print(f"  ✅ Nombre           : {user.get('fullName', 'N/A')}")
        print(f"  ✅ Organización     : {gis.properties.get('name', 'N/A')}")
        return True
    except Exception as e:
        print(f"  ❌ Error ArcGIS: {e}")
        print("     → Verifica usuario/contraseña en .env")
        return False


def test_list_content():
    """Lista las primeras 5 capas del usuario en ArcGIS."""
    print("\n" + "="*50)
    print("TEST 3: Listar contenido ArcGIS")
    print("="*50)
    try:
        from tools.arcgis_tools import list_content
        import json
        result = json.loads(list_content.invoke({"max_items": 5}))
        print(f"  ✅ Items encontrados: {result['total']}")
        for item in result.get("items", [])[:3]:
            print(f"     - [{item['tipo']}] {item['titulo']}")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_db_connection():
    """Verifica la conexión a PostgreSQL + PostGIS."""
    print("\n" + "="*50)
    print("TEST 4: Conexión PostgreSQL + PostGIS")
    print("="*50)
    try:
        from db.connection import test_connection
        result = test_connection()
        if result["status"] == "ok":
            print(f"  ✅ PostgreSQL: {result['postgresql']}")
            print(f"  ✅ PostGIS   : {result['postgis']}")
            return True
        else:
            print(f"  ❌ Error DB: {result['message']}")
            print("     → Verifica que PostgreSQL con PostGIS esté corriendo")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_create_tables():
    """Crea las tablas en la base de datos."""
    print("\n" + "="*50)
    print("TEST 5: Crear tablas en la base de datos")
    print("="*50)
    try:
        from db.connection import engine, Base
        from db import models  # noqa: importa los modelos para registrarlos
        Base.metadata.create_all(bind=engine)
        print("  ✅ Tablas creadas: sessions, agent_logs, geo_results")
        return True
    except Exception as e:
        print(f"  ❌ Error creando tablas: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 VERIFICACIÓN DEL SISTEMA MULTI-AGENTE GEOESPACIAL")
    print("=" * 50)

    results = {
        "settings"        : test_settings(),
        "arcgis"          : test_arcgis_connection(),
        "arcgis_content"  : test_list_content(),
        "database"        : test_db_connection(),
        "tables"          : test_create_tables(),
    }

    print("\n" + "="*50)
    print("RESUMEN")
    print("="*50)
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
    print(f"\n  {passed}/{total} tests pasaron")

    if passed == total:
        print("\n  🎉 Todo listo — puedes avanzar al Paso 2 (GeoAgent)")
    else:
        print("\n  ⚠️  Corrige los errores antes de continuar")
    print()
