"""
tools/arcgis_tools.py
──────────────────────
Herramientas de ArcGIS Online, diseñadas para SLM.

Principios
──────────
1. Argumentos tipados (layer_url: str, max_records: int …). El modelo usa
   tool calling nativo; nunca tiene que escribir un JSON dentro de un string.
2. Salidas compactas. Al LLM le llegan conteos y atributos recortados; las
   geometrías van por el canal lateral (tools/result_store.py) al mapa.
3. Errores que enseñan. Si el campo no existe, el error trae la lista de
   campos válidos para que el modelo corrija en el siguiente intento.
4. Tolerancia: se aceptan alias de campos ("Altura total") y URLs de
   FeatureServer sin el número de subcapa.
"""
import json
import logging
import re
import unicodedata
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool
from arcgis.gis import GIS
from arcgis.features import FeatureLayer
from config.settings import get_settings
from tools import result_store

logger = logging.getLogger(__name__)
settings = get_settings()

_gis_instance: Optional[GIS] = None

MAX_RECORDS = 50        # features que se piden a ArcGIS por consulta
MAX_ROWS_TO_LLM = 8     # filas de atributos que ve el LLM
MAX_ATTRS_PER_ROW = 12  # campos por fila que ve el LLM
SYSTEM_FIELDS = {"globalid", "creator", "editor", "creationdate", "editdate",
                 "shape__area", "shape__length", "shape_area", "shape_length"}


# ══════════════════════════════════════════════════════════
# CONEXIÓN Y UTILIDADES
# ══════════════════════════════════════════════════════════

def get_gis_connection() -> GIS:
    global _gis_instance
    if _gis_instance is None:
        if not settings.arcgis_configured:
            raise ValueError(
                "Faltan credenciales de ArcGIS. Agregá ARCGIS_USERNAME y "
                "ARCGIS_PASSWORD a tu archivo .env y reiniciá el backend."
            )
        logger.info("Conectando a ArcGIS Online...")
        _gis_instance = GIS(
            url=settings.arcgis_url,
            username=settings.arcgis_username,
            password=settings.arcgis_password,
        )
        logger.info(f"Conectado como: {_gis_instance.properties.user.username}")
    return _gis_instance


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _err(msg: str, **extra) -> str:
    return _dump({"status": "error", "mensaje": msg, **extra})


def _clean(s: Any) -> str:
    return str(s or "").strip().strip('"').strip("'").strip()


def clean_where(where: Any) -> str:
    """
    Limpia un filtro SQL SIN romper sus comillas. _clean() no sirve aquí: quita la
    comilla final de "gravedad = 'alta'" y ArcGIS responde "'where' parameter is invalid".
    Solo se quitan las comillas que envuelven TODO el filtro (un SLM a veces lo
    devuelve como "'categoria = basura'" o "\"x = 'a'\"").
    """
    w = str(where or "").strip()
    for q in ('"', "'"):
        if len(w) >= 2 and w[0] == q and w[-1] == q and q not in w[1:-1]:
            w = w[1:-1].strip()
    return w or "1=1"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def normalize_layer_url(url: str) -> str:
    """'.../FeatureServer' → '.../FeatureServer/0'. Quita /query y barras finales."""
    url = _clean(url).rstrip("/")
    url = re.sub(r"/query$", "", url, flags=re.IGNORECASE)
    if re.search(r"/(FeatureServer|MapServer)$", url, re.IGNORECASE):
        url += "/0"
    return url


def get_layer(layer_url: str) -> FeatureLayer:
    url = normalize_layer_url(layer_url)
    if not url.lower().startswith("http"):
        raise ValueError(
            f"'{layer_url}' no es una URL de capa. Usá search_layers para obtener la URL "
            f"(termina en /FeatureServer/0)."
        )
    return FeatureLayer(url, gis=get_gis_connection())


def _fields(layer: FeatureLayer) -> List[Dict[str, str]]:
    return [{"nombre": f.get("name"), "alias": f.get("alias"), "tipo": f.get("type")}
            for f in layer.properties.get("fields", [])]


_TIPOS = {"esriFieldTypeString": "texto", "esriFieldTypeDouble": "número", "esriFieldTypeInteger": "entero",
          "esriFieldTypeSmallInteger": "entero", "esriFieldTypeBigInteger": "entero",
          "esriFieldTypeSingle": "número", "esriFieldTypeDate": "fecha", "esriFieldTypeOID": "id"}
MAX_FIELDS_TO_LLM = 80


def _fields_brief(layer: FeatureLayer) -> List[str]:
    """
    Campos en formato compacto para el LLM: "NOMBRE (alias): tipo".
    Una capa de 200 campos en JSON completo ocupa ~3.000 tokens, casi la mitad del
    contexto de un SLM; así ocupa menos de un tercio y se recorta a 80 campos.
    """
    out = []
    for f in _fields(layer):
        if _norm(f["nombre"]) in SYSTEM_FIELDS:
            continue
        alias = f["alias"] if f["alias"] and _norm(f["alias"]) != _norm(f["nombre"]) else ""
        out.append(f"{f['nombre']}{f' ({alias})' if alias else ''}: {_TIPOS.get(f['tipo'], f['tipo'])}")
    if len(out) > MAX_FIELDS_TO_LLM:
        out = out[:MAX_FIELDS_TO_LLM] + [f"… y {len(out) - MAX_FIELDS_TO_LLM} campos más"]
    return out


def resolve_field(layer: FeatureLayer, field_name: str) -> str:
    """
    Devuelve el nombre real del campo aceptando el nombre, el alias, mayúsculas
    distintas o tildes. Si no existe, levanta un error con los campos válidos.
    """
    wanted = _norm(field_name)
    fields = layer.properties.get("fields", [])
    for f in fields:
        if _norm(f.get("name", "")) == wanted:
            return f["name"]
    for f in fields:
        if _norm(f.get("alias", "")) == wanted:
            return f["name"]
    import difflib
    candidatos = [f for f in fields if _norm(f["name"]) not in SYSTEM_FIELDS]
    # Los parecidos primero; en capas de ~200 campos la lista completa ahoga al SLM
    parecidos = set(difflib.get_close_matches(wanted, [_norm(f["name"]) for f in candidatos], n=5, cutoff=0.5))
    candidatos.sort(key=lambda f: _norm(f["name"]) not in parecidos)
    etiquetas = [f"{f['name']} ({f.get('alias')})" if f.get("alias") and f.get("alias") != f["name"]
                 else f["name"] for f in candidatos]
    if len(etiquetas) > 40:
        etiquetas = etiquetas[:40] + [f"… y {len(etiquetas) - 40} más (usa get_layer_info)"]
    raise ValueError(f"El campo '{field_name}' no existe en la capa. Campos válidos: {', '.join(etiquetas)}")


def _compact_rows(features) -> List[Dict]:
    rows = []
    for f in features[:MAX_ROWS_TO_LLM]:
        attrs = {k: v for k, v in (f.attributes or {}).items()
                 if _norm(k) not in SYSTEM_FIELDS and v not in (None, "")}
        rows.append(dict(list(attrs.items())[:MAX_ATTRS_PER_ROW]))
    return rows


def _store(layer: FeatureLayer, features) -> None:
    title = layer.properties.get("name", "")
    result_store.push(layer.url, [{"attributes": f.attributes, "geometry": f.geometry}
                                  for f in features if f.geometry], layer_name=title)


# ══════════════════════════════════════════════════════════
# HERRAMIENTAS
# ══════════════════════════════════════════════════════════

@tool
def connect_to_arcgis() -> str:
    """Verifica la conexión con ArcGIS Online y devuelve usuario y organización."""
    try:
        gis = get_gis_connection()
        user = gis.properties.user
        return _dump({"status": "conectado", "usuario": user.username,
                      "nombre": user.get("fullName", ""),
                      "organizacion": gis.properties.get("name", "")})
    except Exception as e:
        return _err(str(e))


@tool
def list_content(max_items: int = 10) -> str:
    """Lista los items (capas, mapas) del usuario en ArcGIS Online."""
    try:
        gis = get_gis_connection()
        items = gis.content.search(query=f"owner:{gis.properties.user.username}",
                                   max_items=max(1, min(int(max_items), 50)))
        return _dump({"total": len(items), "items": [
            {"id": i.id, "titulo": i.title, "tipo": i.type, "url": i.url or ""} for i in items]})
    except Exception as e:
        return _err(str(e))


@tool
def search_layers(keyword: str, max_items: int = 8) -> str:
    """
    Busca Feature Layers en ArcGIS Online por palabra clave.
    Devuelve título, item_id y la URL de cada subcapa (lo que piden las demás herramientas).
    """
    try:
        gis = get_gis_connection()
        items = gis.content.search(query=_clean(keyword), item_type="Feature Service",
                                   max_items=max(1, min(int(max_items), 20)))
        capas = []
        for it in items:
            sub = []
            try:
                sub = [{"url": l.url, "nombre": l.properties.get("name", "")}
                       for l in (it.layers or [])[:5]]
            except Exception:
                pass
            capas.append({"titulo": it.title, "item_id": it.id, "propietario": it.owner,
                          "subcapas": sub or [{"url": f"{it.url}/0" if it.url else "", "nombre": ""}]})
        return _dump({"busqueda": keyword, "total": len(capas), "capas": capas})
    except Exception as e:
        return _err(str(e))


@tool
def get_layer_info(layer_url: str) -> str:
    """
    Describe un Feature Layer: geometría, total de registros y sus campos (nombre, alias, tipo).
    Úsala antes de filtrar o calcular estadísticas para conocer los nombres reales de los campos.
    """
    try:
        layer = get_layer(layer_url)
        props = layer.properties
        info = {
            "url": layer.url,
            "nombre": props.get("name", ""),
            "geometria": props.get("geometryType", ""),
            "total_registros": layer.query(return_count_only=True),
            "campos": _fields_brief(layer),
        }
        # Los valores de los campos categóricos: sin ellos el modelo no sabe que
        # "reportes de seguridad" es categoria = 'seguridad' (visto en las trazas)
        from tools.vector_tools import indexed_values
        valores = indexed_values(layer.url) or indexed_values(normalize_layer_url(layer_url))
        if valores:
            info["valores"] = valores
        return _dump(info)
    except Exception as e:
        return _err(str(e))


@tool
def count_features(layer_url: str, where: str = "1=1") -> str:
    """Cuenta los registros de un Feature Layer, opcionalmente con un filtro SQL (where)."""
    try:
        layer = get_layer(layer_url)
        n = layer.query(where=clean_where(where), return_count_only=True)
        return _dump({"url": layer.url, "filtro": where, "total": n})
    except Exception as e:
        return _err(str(e))


@tool
def query_features(layer_url: str, where: str = "1=1", max_records: int = 10) -> str:
    """
    Consulta registros de un Feature Layer con un filtro SQL (ej. "especie = 'Eucalipto'").
    Devuelve el total que cumple el filtro y algunos registros de muestra.
    Los resultados se dibujan en el mapa automáticamente.
    """
    try:
        layer = get_layer(layer_url)
        where = clean_where(where)
        n = max(1, min(int(max_records), MAX_RECORDS))
        total = layer.query(where=where, return_count_only=True)
        fs = layer.query(where=where, out_fields="*", return_geometry=True, out_sr=4326,
                         result_record_count=n, return_all_records=False)
        _store(layer, fs.features)
        return _dump({"url": layer.url, "filtro": where, "total_que_cumplen": total,
                      "devueltos": len(fs.features), "muestra": _compact_rows(fs.features)})
    except Exception as e:
        return _err(str(e), ayuda="Revisá los nombres de campo con get_layer_info.")


@tool
def query_features_near(layer_url: str, lon: float, lat: float, distance_m: float = 500,
                        where: str = "1=1", max_records: int = 20) -> str:
    """
    Consulta espacial: registros de un Feature Layer a menos de distance_m metros
    de un punto (lon, lat en WGS84). Ejemplo Quito centro: lon=-78.4678, lat=-0.1807.
    Los resultados se dibujan en el mapa automáticamente.
    """
    try:
        from arcgis.geometry import Geometry
        from arcgis.geometry.filters import intersects

        layer = get_layer(layer_url)
        point = Geometry({"x": float(lon), "y": float(lat), "spatialReference": {"wkid": 4326}})
        gfilter = intersects(point, sr=4326)
        common = dict(where=clean_where(where), geometry_filter=gfilter,
                      distance=float(distance_m), units="esriSRUnit_Meter")
        total = layer.query(return_count_only=True, **common)
        fs = layer.query(out_fields="*", return_geometry=True, out_sr=4326,
                         result_record_count=max(1, min(int(max_records), MAX_RECORDS)),
                         return_all_records=False, **common)
        _store(layer, fs.features)
        return _dump({"url": layer.url, "punto": {"lon": lon, "lat": lat},
                      "radio_m": distance_m, "total_en_radio": total,
                      "devueltos": len(fs.features), "muestra": _compact_rows(fs.features)})
    except Exception as e:
        return _err(str(e))


GEO_TOOLS = [search_layers, get_layer_info, count_features, query_features,
             query_features_near, list_content, connect_to_arcgis]
