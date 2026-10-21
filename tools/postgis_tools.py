"""
tools/postgis_tools.py
───────────────────────
Consultas espaciales sobre PostgreSQL + PostGIS (resultados ya guardados).

La tabla geo_results se llena sola: cada feature que un agente trae de
ArcGIS Online durante una conversación se persiste al final de la petición
(ver agents/dispatch.py). Así el QueryAgent puede responder "¿cuáles de los
árboles que consulté están a 300 m de este punto?".

Argumentos tipados, geometrías al mapa por el canal lateral.
"""
import json
import logging
import re

from langchain_core.tools import tool
from sqlalchemy import text

from db.connection import SessionLocal
from tools import result_store

logger = logging.getLogger(__name__)


def _dump(o) -> str:
    return json.dumps(o, ensure_ascii=False, default=str)


def _push_geojson_rows(rows) -> None:
    """Las filas traen geojson; se mandan al mapa como features ArcGIS-like."""
    feats = []
    for r in rows:
        g = r.geojson
        if isinstance(g, str):
            g = json.loads(g)
        if g and g.get("type") == "Point":
            feats.append({"attributes": r.attributes or {},
                          "geometry": {"x": g["coordinates"][0], "y": g["coordinates"][1]}})
    if feats:
        result_store.push("postgis://geo_results", feats, layer_name="PostGIS")


@tool
def count_by_layer() -> str:
    """Cuántos registros hay guardados en PostGIS por cada capa."""
    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT layer_name, COUNT(*) AS total, COUNT(geometry) AS con_geometria,
                   MAX(created_at)::text AS ultima_vez
            FROM geo_results GROUP BY layer_name ORDER BY total DESC
        """)).fetchall()
        return _dump({"total_capas": len(rows), "capas": [dict(r._mapping) for r in rows]})
    except Exception as e:
        return _dump({"status": "error", "mensaje": str(e)})
    finally:
        db.close()


@tool
def query_geo_results(layer_name: str = "", limit: int = 10) -> str:
    """Lista registros guardados en PostGIS, opcionalmente filtrando por nombre de capa."""
    db = SessionLocal()
    try:
        cond, params = "", {"limit": max(1, min(int(limit), 50))}
        if layer_name:
            cond, params["ln"] = "WHERE layer_name ILIKE :ln", f"%{layer_name}%"
        rows = db.execute(text(f"""
            SELECT id, layer_name, attributes, ST_AsGeoJSON(geometry)::json AS geojson
            FROM geo_results {cond} ORDER BY created_at DESC LIMIT :limit
        """), params).fetchall()
        _push_geojson_rows(rows)
        return _dump({"total": len(rows), "registros": [
            {"id": r.id, "capa": r.layer_name, "atributos": r.attributes} for r in rows[:10]]})
    except Exception as e:
        return _dump({"status": "error", "mensaje": str(e)})
    finally:
        db.close()


@tool
def spatial_buffer(lon: float, lat: float, radio_metros: float = 1000, layer_name: str = "") -> str:
    """
    Registros guardados en PostGIS a menos de radio_metros de un punto (lon, lat WGS84).
    Ejemplo Quito centro: lon=-78.4678, lat=-0.1807.
    """
    db = SessionLocal()
    try:
        params = {"lon": float(lon), "lat": float(lat), "radio": float(radio_metros)}
        cond = ""
        if layer_name:
            cond, params["ln"] = "AND layer_name ILIKE :ln", f"%{layer_name}%"
        rows = db.execute(text(f"""
            SELECT id, layer_name, attributes, ST_AsGeoJSON(geometry)::json AS geojson,
                   ST_Distance(geometry::geography,
                               ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) AS d
            FROM geo_results
            WHERE geometry IS NOT NULL
              AND ST_DWithin(geometry::geography,
                             ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography, :radio)
              {cond}
            ORDER BY d ASC LIMIT 50
        """), params).fetchall()
        _push_geojson_rows(rows)
        return _dump({"punto": {"lon": lon, "lat": lat}, "radio_metros": radio_metros,
                      "total": len(rows),
                      "mas_cercanos": [{"capa": r.layer_name, "distancia_m": round(float(r.d), 1),
                                        "atributos": r.attributes} for r in rows[:8]]})
    except Exception as e:
        return _dump({"status": "error", "mensaje": str(e)})
    finally:
        db.close()


@tool
def find_nearest(lon: float, lat: float, layer_name: str = "") -> str:
    """El registro guardado en PostGIS más cercano a un punto (lon, lat WGS84)."""
    db = SessionLocal()
    try:
        params = {"lon": float(lon), "lat": float(lat)}
        cond = ""
        if layer_name:
            cond, params["ln"] = "AND layer_name ILIKE :ln", f"%{layer_name}%"
        r = db.execute(text(f"""
            SELECT id, layer_name, attributes, ST_AsGeoJSON(geometry)::json AS geojson,
                   ST_Distance(geometry::geography,
                               ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography) AS d
            FROM geo_results WHERE geometry IS NOT NULL {cond}
            ORDER BY d ASC LIMIT 1
        """), params).fetchone()
        if not r:
            return _dump({"status": "sin_resultados",
                          "mensaje": "No hay registros guardados con geometría"})
        _push_geojson_rows([r])
        return _dump({"capa": r.layer_name, "distancia_m": round(float(r.d), 1),
                      "atributos": r.attributes})
    except Exception as e:
        return _dump({"status": "error", "mensaje": str(e)})
    finally:
        db.close()


@tool
def run_sql_spatial(sql_query: str) -> str:
    """
    Consulta SQL de solo lectura sobre PostGIS (tablas: geo_results, sessions, agent_logs).
    Una sola sentencia SELECT. Úsala solo si ninguna otra herramienta sirve.
    """
    db = SessionLocal()
    try:
        sql_query = str(sql_query).strip().rstrip(";").strip()

        # Seguridad: una sola sentencia SELECT/WITH. El SQL lo escribe el LLM y
        # el LLM lee texto de ArcGIS (títulos, descripciones): eso es una vía de
        # prompt injection. "SELECT 1; DROP TABLE sessions" pasaría un simple
        # startswith("SELECT"), porque psycopg2 ejecuta varias sentencias.
        if ";" in sql_query:
            return _dump({"status": "error", "mensaje": "Solo se permite una sentencia"})
        if not re.match(r"^(SELECT|WITH)\b", sql_query, re.IGNORECASE):
            return _dump({"status": "error", "mensaje": "Solo se permiten consultas SELECT"})

        # Segunda barrera: transacción de solo lectura, con timeout y límite de filas
        db.execute(text("SET TRANSACTION READ ONLY"))
        db.execute(text("SET LOCAL statement_timeout = 10000"))
        result = db.execute(text(f"SELECT * FROM ({sql_query}) AS q LIMIT 200"))
        rows = result.fetchall()
        columns = list(result.keys())
        data = [dict(zip(columns, row)) for row in rows]
        return _dump({"columnas": columns, "total": len(data), "datos": data[:30]})
    except Exception as e:
        db.rollback()
        return _dump({"status": "error", "mensaje": str(e)})
    finally:
        db.close()


QUERY_TOOLS = [count_by_layer, query_geo_results, spatial_buffer, find_nearest, run_sql_spatial]
