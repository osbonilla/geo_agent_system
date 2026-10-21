"""
tools/analytics_tools.py
─────────────────────────
Estadísticas sobre Feature Layers de ArcGIS Online, diseñadas para SLM.

Todas las estadísticas las calcula el servidor de ArcGIS (outStatistics):
el modelo nunca suma ni promedia, solo elige capa, campo y operación.
Eso es deliberado: la aritmética es justo donde un SLM se equivoca.
"""
import logging
from langchain_core.tools import tool

from tools.arcgis_tools import (
    get_layer, resolve_field, clean_where, _dump, _err, _clean, _compact_rows, _store, _fields,
    _norm, SYSTEM_FIELDS,
)

logger = logging.getLogger(__name__)

NUMERIC = {"esriFieldTypeDouble", "esriFieldTypeInteger", "esriFieldTypeSingle",
           "esriFieldTypeSmallInteger", "esriFieldTypeBigInteger"}


def _r(v):
    return round(v, 2) if isinstance(v, float) else v


@tool
def field_statistics(layer_url: str, field_name: str, where: str = "1=1") -> str:
    """
    Mínimo, máximo, promedio, suma y conteo de un campo NUMÉRICO.
    Úsala para: máximo, mínimo, promedio, total de un campo.
    """
    try:
        layer = get_layer(layer_url)
        field = resolve_field(layer, field_name)
        stats = [{"statisticType": t, "onStatisticField": field, "outStatisticFieldName": f"v_{t}"}
                 for t in ("min", "max", "avg", "sum", "count")]
        fs = layer.query(where=clean_where(where), out_statistics=stats)
        if not fs.features:
            return _dump({"status": "sin_datos"})
        a = {k.lower(): v for k, v in fs.features[0].attributes.items()}
        return _dump({"campo": field, "filtro": where,
                      "minimo": _r(a.get("v_min")), "maximo": _r(a.get("v_max")),
                      "promedio": _r(a.get("v_avg")), "suma": _r(a.get("v_sum")),
                      "conteo": a.get("v_count")})
    except Exception as e:
        return _err(str(e))


@tool
def top_n_features(layer_url: str, field_name: str, n: int = 5, order: str = "desc") -> str:
    """
    Los N registros con el valor más alto (order="desc") o más bajo (order="asc") de un campo.
    Úsala para: top 5, los más altos, los menores, ranking. Se dibujan en el mapa.
    """
    try:
        layer = get_layer(layer_url)
        field = resolve_field(layer, field_name)
        order = "ASC" if str(order).lower().startswith("asc") else "DESC"
        n = max(1, min(int(n), 20))
        fs = layer.query(where=f"{field} IS NOT NULL", out_fields="*",
                         order_by_fields=f"{field} {order}", result_record_count=n,
                         return_all_records=False, return_geometry=True, out_sr=4326)
        _store(layer, fs.features)
        return _dump({"campo": field, "orden": order, "n": len(fs.features),
                      "registros": _compact_rows(fs.features)})
    except Exception as e:
        return _err(str(e))


@tool
def count_by_category(layer_url: str, field_name: str, top: int = 15) -> str:
    """
    Conteo de registros agrupados por un campo categórico (distribución).
    Úsala para: ¿cuántos hay por especie?, distribución por tipo, ¿qué valores hay y cuántos?
    """
    try:
        layer = get_layer(layer_url)
        field = resolve_field(layer, field_name)
        fs = layer.query(
            where="1=1", group_by_fields_for_statistics=field,
            out_statistics=[{"statisticType": "count", "onStatisticField": field,
                             "outStatisticFieldName": "conteo"}],
            order_by_fields="conteo DESC",
        )
        rows = []
        for f in fs.features:
            a = {k.lower(): v for k, v in f.attributes.items()}
            rows.append({"categoria": a.get(field.lower(), "Sin valor"), "conteo": a.get("conteo", 0)})
        top = max(1, min(int(top), 50))
        return _dump({"campo": field, "total_categorias": len(rows), "categorias": rows[:top],
                      "nota": f"se muestran {min(top, len(rows))} de {len(rows)}" if len(rows) > top else ""})
    except Exception as e:
        return _err(str(e))


@tool
def layer_summary(layer_url: str) -> str:
    """
    Resumen de una capa: total de registros, geometría, estadísticas de los campos
    numéricos y lista de campos de texto. Úsala para "analiza / resume esta capa".
    """
    try:
        layer = get_layer(layer_url)
        props = layer.properties
        campos = [f for f in _fields(layer) if _norm(f["nombre"]) not in SYSTEM_FIELDS
                  and f["nombre"].lower() not in ("objectid", "fid")]
        numericos = [f for f in campos if f["tipo"] in NUMERIC][:5]
        resumen = []
        for c in numericos:
            try:
                stats = [{"statisticType": t, "onStatisticField": c["nombre"],
                          "outStatisticFieldName": f"v_{t}"} for t in ("min", "max", "avg")]
                a = {k.lower(): v for k, v in
                     layer.query(where="1=1", out_statistics=stats).features[0].attributes.items()}
                resumen.append({"campo": c["nombre"], "alias": c["alias"], "min": _r(a.get("v_min")),
                                "max": _r(a.get("v_max")), "promedio": _r(a.get("v_avg"))})
            except Exception:
                continue
        return _dump({
            "url": layer.url, "nombre": props.get("name", ""),
            "total_registros": layer.query(return_count_only=True),
            "geometria": props.get("geometryType", ""),
            "campos_numericos": resumen,
            "campos_texto": [{"nombre": c["nombre"], "alias": c["alias"]}
                             for c in campos if c["tipo"] == "esriFieldTypeString"][:10],
        })
    except Exception as e:
        return _err(str(e))


ANALYTICS_TOOLS = [field_statistics, top_n_features, count_by_category, layer_summary]
