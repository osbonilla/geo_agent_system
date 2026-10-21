"""
tools/spatial_tools.py
───────────────────────
Análisis espacial ENTRE capas (y respecto a un lugar), diseñado para SLM.

Qué agrega al sistema
─────────────────────
Las herramientas de GeoAgent y AnalyticsAgent trabajan sobre UNA capa.
Estas relacionan capas por su ubicación:

  count_points_in_polygons  cuántos puntos caen en cada polígono
                            (accidentes por parroquia, denuncias por parroquia)
  overlay_polygons          qué polígonos cruza otra capa y en qué porcentaje
                            (parroquias en zona de amenaza volcánica)
  count_points_near         cuántos puntos de una capa están a menos de N m de otra
                            (reportes de seguridad cerca de accidentes)
  nearest_features          lo más cercano a un punto / lo que está a menos de N km
  locate_point              en qué polígono cae un punto (¿en qué parroquia está?)
  geocode_place             nombre de lugar → coordenadas

Por qué se calcula aquí y no con los servicios de análisis de ArcGIS
────────────────────────────────────────────────────────────────────
Los servicios de análisis de ArcGIS Online (Aggregate Points, Overlay
Layers, Find Nearest…) consumen créditos y tardan minutos. Una consulta
(query) que devuelve geometrías no consume créditos. Se descargan las
features en WGS84, se calcula con Shapely y al LLM le llega solo el
resultado compacto; las geometrías van al mapa por el canal lateral
(tools/result_store.py). Como en analytics_tools, el modelo nunca calcula:
solo elige capas, filtros y herramienta.

Distancias y áreas
──────────────────
Se proyectan las coordenadas a metros con una proyección equirectangular
local (x · cos(lat₀), y), con lat₀ en el centro de los datos. En Ecuador,
atravesado por la línea equinoccial, el error es menor al 0,5 %, sin
depender de pyproj.
"""
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely
from langchain_core.tools import tool
from shapely.geometry import LinearRing, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.strtree import STRtree

from tools.arcgis_tools import (
    get_layer, get_gis_connection, resolve_field, clean_where, _dump, _err, _clean, _norm, SYSTEM_FIELDS,
)
from tools import result_store

logger = logging.getLogger(__name__)

M_POR_GRADO_LAT = 110_574.0
M_POR_GRADO_LON = 111_320.0
MAX_FILAS_LLM = 15          # filas de resultado que ve el LLM
MAX_AL_MAPA = 500           # features que se envían al mapa por herramienta
CACHE_TTL_S = 300           # las capas se reutilizan 5 min (útil en evaluación y en planes de varios pasos)
CACHE_MAX = 24

# Campos que suelen tener el nombre del polígono, en orden de preferencia
NOMBRE_PISTAS = ("nombre", "name", "despar", "despro", "descan", "parroquia", "provincia",
                 "canton", "titulo", "volcan", "label")


# ══════════════════════════════════════════════════════════
# DESCARGA DE FEATURES (con caché)
# ══════════════════════════════════════════════════════════

_cache: Dict[Tuple[str, str, str], Tuple[float, list]] = {}


def clear_cache() -> None:
    _cache.clear()


def _oid_field(layer) -> str:
    props = layer.properties
    if props.get("objectIdField"):
        return props["objectIdField"]
    for f in props.get("fields", []):
        if f.get("type") == "esriFieldTypeOID":
            return f["name"]
    return "OBJECTID"


def _fetch(layer, where: str = "1=1", campos: Optional[List[Optional[str]]] = None) -> List[dict]:
    """
    Features de la capa que cumplen el filtro: [{"attributes", "geometry"}] en WGS84.
    Solo se piden los campos necesarios: la capa de accidentes tiene ~200 campos y
    traerlos todos multiplica el tamaño de la descarga sin aportar nada al cálculo.
    """
    where = clean_where(where)
    out_fields = ",".join(dict.fromkeys([_oid_field(layer)] + [c for c in (campos or []) if c]))
    key = (layer.url, where, out_fields)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    fs = layer.query(where=where, out_fields=out_fields, return_geometry=True, out_sr=4326,
                     return_all_records=True)
    feats = [{"attributes": dict(f.attributes or {}), "geometry": f.geometry}
             for f in fs.features if f.geometry]
    if len(_cache) >= CACHE_MAX:
        _cache.pop(min(_cache, key=lambda k: _cache[k][0]))
    _cache[key] = (time.time(), feats)
    return feats


# ══════════════════════════════════════════════════════════
# GEOMETRÍAS
# ══════════════════════════════════════════════════════════

def esri_to_shapely(g: Optional[dict]):
    """Geometría JSON de ArcGIS → Shapely. En polígonos, los anillos horarios son
    exteriores y los antihorarios, huecos (convención de Esri)."""
    if not g:
        return None
    if "x" in g:
        return None if g.get("x") is None else Point(g["x"], g["y"])
    if "points" in g:
        return MultiPoint(g["points"]) if g["points"] else None
    if "paths" in g:
        paths = [p for p in g["paths"] if len(p) >= 2]
        if not paths:
            return None
        return LineString(paths[0]) if len(paths) == 1 else MultiLineString(paths)
    if "rings" in g:
        exteriores, huecos = [], []
        for r in g["rings"]:
            if len(r) < 4:
                continue
            (huecos if LinearRing(r).is_ccw else exteriores).append(r)
        if not exteriores:                       # anillos con la orientación invertida
            exteriores, huecos = huecos, []
        partes = [[e, []] for e in exteriores]
        cascarones = [Polygon(e) for e in exteriores]
        for h in huecos:
            p = Point(h[0])
            for i, c in enumerate(cascarones):
                if c.contains(p):
                    partes[i][1].append(h)
                    break
        polys = [Polygon(e, hs) for e, hs in partes]
        geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        if not geom.is_valid:
            geom = shapely.make_valid(geom)
            if geom.geom_type == "GeometryCollection":
                geom = MultiPolygon([p for p in geom.geoms if p.geom_type == "Polygon"]
                                    + [q for p in geom.geoms if p.geom_type == "MultiPolygon" for q in p.geoms])
        return geom
    return None


def _a_metros(geom, lat0: float):
    """Proyección equirectangular local: grados → metros."""
    k = np.array([M_POR_GRADO_LON * math.cos(math.radians(lat0)), M_POR_GRADO_LAT])
    return shapely.transform(geom, lambda c: c * k)


def _lat_centro(*geoms) -> float:
    lats = [g.centroid.y for g in geoms if g is not None and not g.is_empty]
    return float(np.mean(lats)) if lats else 0.0


def _preparar(feats: List[dict]) -> List[Tuple[dict, object]]:
    out = []
    for f in feats:
        try:
            g = esri_to_shapely(f["geometry"])
        except Exception as e:                   # una geometría corrupta no tumba el análisis
            logger.debug(f"Geometría descartada: {e}")
            continue
        if g is not None and not g.is_empty:
            out.append((f, g))
    return out


# ══════════════════════════════════════════════════════════
# CAMPOS Y NOMBRES
# ══════════════════════════════════════════════════════════

def _campo_nombre(layer, name_field: str = "") -> Optional[str]:
    """Campo que identifica cada feature. Si el modelo no lo da, se adivina."""
    if _clean(name_field):
        return resolve_field(layer, name_field)
    campos = [f for f in layer.properties.get("fields", [])
              if f.get("type") == "esriFieldTypeString" and _norm(f.get("name", "")) not in SYSTEM_FIELDS]
    for pista in NOMBRE_PISTAS:
        for f in campos:
            if pista in _norm(f.get("name", "")) or pista in _norm(f.get("alias", "")):
                return f["name"]
    return campos[0]["name"] if campos else None


def _tiene_campo(layer, nombre: str) -> bool:
    try:
        resolve_field(layer, nombre)
        return True
    except ValueError:
        return False


def _valores_del_filtro(layer, where: str, n: int = 15) -> dict:
    """Si un filtro de texto no devuelve nada, muestra los valores más frecuentes de ese campo:
    el modelo ve que TIPO_DE_SINIESTRO no tiene 'Fallecido' y puede corregir."""
    import re
    m = re.match(r"\s*\(?\s*([A-Za-z_][\w]*)\s*(=|<>|!=|\bIN\b|\bLIKE\b)", clean_where(where), re.IGNORECASE)
    if not m:
        return {}
    try:
        campo = resolve_field(layer, m.group(1))
        fs = layer.query(where="1=1", group_by_fields_for_statistics=campo,
                         out_statistics=[{"statisticType": "count", "onStatisticField": campo,
                                          "outStatisticFieldName": "conteo"}],
                         order_by_fields="conteo DESC")
        vals = [str({k.lower(): v for k, v in f.attributes.items()}.get(campo.lower())) for f in fs.features]
        return {f"valores_de_{campo}": [v for v in vals if v not in ("None", "")][:n]}
    except Exception:
        return {}


def _nombre(attrs: dict, campo: Optional[str], fallback: str) -> str:
    v = attrs.get(campo) if campo else None
    return str(v).strip() if v not in (None, "") else fallback


def _oid(attrs: dict):
    for k in ("OBJECTID", "objectid", "FID", "fid", "ObjectID"):
        if k in attrs:
            return attrs[k]
    return None


def _coincide(nombre: str, buscado: str) -> bool:
    """Coincidencia por nombre sin tildes ni mayúsculas ('la concepcion' ~ 'LA CONCEPCIÓN')."""
    b = _norm(buscado)
    return bool(b) and b in _norm(nombre)


def _filtrar_por_nombre(pares, campo, buscado: str):
    if not _clean(buscado):
        return pares
    return [(f, g) for f, g in pares if _coincide(_nombre(f["attributes"], campo, ""), buscado)]


def _nombres_disponibles(pares, campo, n: int = 15) -> List[str]:
    return sorted({_nombre(f["attributes"], campo, "") for f, _ in pares} - {""})[:n]


def _al_mapa(layer, feats: List[dict], extra: Optional[List[dict]] = None) -> None:
    """Envía features al mapa (y a PostGIS) sin pasar por el LLM."""
    nombre = layer.properties.get("name", "")
    items = []
    for i, f in enumerate(feats[:MAX_AL_MAPA]):
        attrs = {**f["attributes"], **(extra[i] if extra else {})}
        items.append({"attributes": attrs, "geometry": f["geometry"]})
    result_store.push(layer.url, items, layer_name=nombre)


def _r(v, d=2):
    return round(float(v), d)


# ══════════════════════════════════════════════════════════
# HERRAMIENTAS
# ══════════════════════════════════════════════════════════

@tool
def geocode_place(place: str) -> str:
    """
    Convierte el nombre de un lugar de Ecuador en coordenadas (lon, lat en WGS84).
    Úsala antes de nearest_features o locate_point cuando la pregunta nombra un lugar
    ("cerca de Quito", "¿en qué parroquia está el parque La Carolina?").
    """
    try:
        from arcgis.geocoding import geocode
        get_gis_connection()
        # for_storage=False: geocodificar sin guardar el resultado no consume créditos
        res = geocode(_clean(place), max_locations=1, out_sr=4326, source_country="ECU",
                      for_storage=False)
        if not res:
            return _err(f"No se encontró '{place}'. Prueba con un nombre más general (ej. ciudad o parroquia).")
        r = res[0]
        loc = r.get("location", {})
        return _dump({"lugar": place, "lon": _r(loc.get("x"), 6), "lat": _r(loc.get("y"), 6),
                      "direccion": r.get("address", ""), "puntaje": r.get("score")})
    except Exception as e:
        return _err(str(e))


@tool
def locate_point(polygons_url: str, lon: float, lat: float, name_field: str = "") -> str:
    """
    En qué polígono cae un punto (lon, lat): ¿en qué parroquia / provincia / zona está?
    Si el punto no cae en ningún polígono, devuelve el más cercano y su distancia.
    name_field es opcional (campo con el nombre del polígono); si falta, se detecta.
    """
    try:
        layer = get_layer(polygons_url)
        campo = _campo_nombre(layer, name_field)
        pares = _preparar(_fetch(layer, "1=1", [campo]))
        if not pares:
            return _err("La capa de polígonos no tiene geometrías.")
        p = Point(float(lon), float(lat))
        dentro = [(f, g) for f, g in pares if g.covers(p)]
        if dentro:
            _al_mapa(layer, [f for f, _ in dentro])
            return _dump({"punto": {"lon": lon, "lat": lat}, "dentro_de": [
                _nombre(f["attributes"], campo, f"id {_oid(f['attributes'])}") for f, _ in dentro]})
        pm = _a_metros(p, lat)
        f, g = min(pares, key=lambda fg: _a_metros(fg[1], lat).distance(pm))
        _al_mapa(layer, [f])
        return _dump({"punto": {"lon": lon, "lat": lat}, "dentro_de": [],
                      "mas_cercano": _nombre(f["attributes"], campo, f"id {_oid(f['attributes'])}"),
                      "distancia_km": _r(_a_metros(g, lat).distance(pm) / 1000)})
    except Exception as e:
        return _err(str(e))


@tool
def count_points_in_polygons(points_url: str, polygons_url: str, points_where: str = "1=1",
                             polygon_name: str = "", name_field: str = "", group_field: str = "",
                             top: int = 10) -> str:
    """
    Cuenta cuántos puntos de una capa caen dentro de cada polígono de otra capa y
    devuelve el ranking. Úsala para: ¿cuántos accidentes hay por parroquia?, ¿qué
    parroquia tiene más denuncias?, ¿cuántos reportes hay en Quitumbe?
      points_where : filtro SQL sobre los puntos, ej. "intencion_bert = 'denuncia'"
      polygon_name : (opcional) limita a los polígonos cuyo nombre contiene este texto, ej. "Quitumbe"
      group_field  : (opcional) campo de los puntos para desglosar cada polígono, ej. "categoria"
      name_field   : (opcional) campo con el nombre del polígono; si falta, se detecta
    """
    try:
        lp, lg = get_layer(points_url), get_layer(polygons_url)
        if "Polygon" in str(lp.properties.get("geometryType", "")):
            return _err("points_url es una capa de POLÍGONOS. Para relacionar polígonos con polígonos usa "
                        f"overlay_polygons(target_url='{polygons_url}', overlay_url='{points_url}', "
                        f"overlay_where='{clean_where(points_where)}').")
        # Error frecuente de los SLM: pasar en group_field el campo de nombre de los POLÍGONOS
        # (dpa_despar). Se interpreta como name_field en vez de fallar.
        if _clean(group_field) and not _tiene_campo(lp, group_field) and _tiene_campo(lg, group_field):
            name_field, group_field = name_field or group_field, ""
        campo = _campo_nombre(lg, name_field)
        grupo = resolve_field(lp, group_field) if _clean(group_field) else None
        puntos = _preparar(_fetch(lp, points_where, [grupo]))
        todos_poligonos = _preparar(_fetch(lg, "1=1", [campo]))
        poligonos = _filtrar_por_nombre(todos_poligonos, campo, polygon_name)
        if not poligonos:
            return _err(f"Ningún polígono coincide con '{polygon_name}'.",
                        nombres_validos=_nombres_disponibles(todos_poligonos, campo))
        if not puntos:
            return _dump({"filtro_puntos": clean_where(points_where), "total_puntos": 0, "ranking": [],
                          "aviso": "Ningún punto cumple el filtro. Revisa el campo y sus valores con "
                                   "get_layer_info antes de concluir que no hay datos.",
                          **_valores_del_filtro(lp, points_where)})

        arbol = STRtree([g for _, g in poligonos])
        geoms_pt = [g if g.geom_type == "Point" else g.centroid for _, g in puntos]
        idx_pt, idx_pol = arbol.query(geoms_pt, predicate="within")
        asignado = {}
        for i, j in zip(idx_pt.tolist(), idx_pol.tolist()):
            asignado.setdefault(i, j)            # un punto en el borde cuenta una sola vez

        conteo: Dict[int, int] = {}
        desglose: Dict[int, Dict[str, int]] = {}
        for i, j in asignado.items():
            conteo[j] = conteo.get(j, 0) + 1
            if grupo:
                v = str(puntos[i][0]["attributes"].get(grupo) or "sin valor")
                desglose.setdefault(j, {})
                desglose[j][v] = desglose[j].get(v, 0) + 1

        total_dentro = len(asignado)
        orden = sorted(conteo.items(), key=lambda kv: -kv[1])
        top = max(1, min(int(top), 50))
        ranking = []
        for j, n in orden[:top]:
            fila = {"nombre": _nombre(poligonos[j][0]["attributes"], campo, f"polígono {j}"),
                    "conteo": n, "porcentaje": _r(100 * n / total_dentro, 1)}
            if grupo:
                fila["por_" + grupo] = dict(sorted(desglose[j].items(), key=lambda kv: -kv[1])[:6])
            ranking.append(fila)

        _al_mapa(lg, [poligonos[j][0] for j, _ in orden],
                 extra=[{"conteo": n} for _, n in orden])
        return _dump({
            "filtro_puntos": clean_where(points_where),
            "total_puntos": len(puntos), "puntos_dentro": total_dentro,
            "puntos_fuera": len(puntos) - total_dentro,
            "poligonos_con_puntos": len(conteo), "poligonos_evaluados": len(poligonos),
            "campo_nombre": campo, "ranking": ranking,
            "nota": f"se muestran {len(ranking)} de {len(conteo)}" if len(conteo) > len(ranking) else "",
        })
    except Exception as e:
        return _err(str(e), ayuda="Revisa los nombres de campo con get_layer_info.")


@tool
def overlay_polygons(target_url: str, overlay_url: str, overlay_where: str = "1=1",
                     overlay_name: str = "", name_field: str = "", group_field: str = "",
                     min_percent: float = 0.0, order_by: str = "porcentaje", top: int = 15) -> str:
    """
    Superpone dos capas: qué polígonos de target_url toca la capa overlay_url y qué
    porcentaje de su área cubre. Úsala para: ¿qué parroquias están en zona de amenaza
    volcánica?, ¿qué provincias atraviesa el Parque Nacional Yasuní?, ¿qué % de X es Y?
      target_url    : la capa por la que se pregunta (parroquias, provincias)
      overlay_url   : la capa que se superpone (amenaza, áreas protegidas)
      overlay_where : filtro SQL sobre overlay, ej. "tipo = 'Lahares'"
      overlay_name  : (opcional) solo los elementos de overlay cuyo nombre contiene este texto, ej. "Yasuní"
      group_field   : (opcional) campo de overlay para desglosar el porcentaje por categoría
      min_percent   : (opcional) descarta los polígonos cubiertos menos de este porcentaje
      order_by      : "porcentaje" (qué zona tiene MÁS % de SU territorio cubierto, por defecto) o
                      "area" (qué zona contiene la MAYOR PARTE de overlay, ej. ¿en qué provincia
                      está la mayor parte del parque?)
    """
    try:
        lt, lo = get_layer(target_url), get_layer(overlay_url)
        # Mismo error frecuente que en count_points_in_polygons: el campo de nombre de la capa
        # objetivo (dpa_despar) pasado como group_field. Se interpreta como name_field.
        if _clean(group_field) and not _tiene_campo(lo, group_field) and _tiene_campo(lt, group_field):
            name_field, group_field = name_field or group_field, ""
        campo_t = _campo_nombre(lt, name_field)
        campo_o = _campo_nombre(lo)
        grupo = resolve_field(lo, group_field) if _clean(group_field) else None
        objetivos = _preparar(_fetch(lt, "1=1", [campo_t]))
        todos_overlay = _preparar(_fetch(lo, overlay_where, [campo_o, grupo]))
        capa_sup = _filtrar_por_nombre(todos_overlay, campo_o, overlay_name)
        if not capa_sup:
            return _err(f"Ningún elemento de la capa superpuesta coincide con '{overlay_name or overlay_where}'.",
                        nombres_validos=_nombres_disponibles(todos_overlay, campo_o))
        if not objetivos:
            return _err("La capa objetivo no tiene geometrías.")

        lat0 = _lat_centro(*[g for _, g in objetivos[:50]])
        es_area = all(g.geom_type in ("Polygon", "MultiPolygon") for _, g in capa_sup)

        # Overlay agrupado por categoría (o todo junto) y proyectado a metros
        grupos: Dict[str, list] = {}
        for f, g in capa_sup:
            k = str(f["attributes"].get(grupo) or "sin valor") if grupo else "_todo"
            grupos.setdefault(k, []).append(g)
        uniones = {k: _a_metros(shapely.union_all(gs) if es_area else shapely.GeometryCollection(gs), lat0)
                   for k, gs in grupos.items()}
        # Las categorías pueden solaparse: el área cubierta total se mide contra la unión de todas
        union_total = shapely.union_all(list(uniones.values())) if es_area else None
        area_overlay = union_total.area if es_area else 0.0
        arbol = STRtree([_a_metros(g, lat0) for _, g in capa_sup])

        filas, afectados = [], []
        for f, g in objetivos:
            gm = _a_metros(g, lat0)
            candidatos = arbol.query(gm, predicate="intersects")
            if len(candidatos) == 0:
                continue
            area_t = gm.area
            fila = {"nombre": _nombre(f["attributes"], campo_t, f"id {_oid(f['attributes'])}"),
                    "area_km2": _r(area_t / 1e6)}
            if es_area:
                cubierta = gm.intersection(union_total).area
                pct = 100 * cubierta / area_t if area_t else 0.0
                if pct < float(min_percent) or cubierta <= 0:
                    continue
                fila.update({"area_cubierta_km2": _r(cubierta / 1e6), "porcentaje_cubierto": _r(pct, 1),
                             # qué parte de la capa superpuesta cae en esta zona
                             "porcentaje_del_overlay": _r(100 * cubierta / area_overlay, 1) if area_overlay else 0.0})
                if grupo:
                    fila["por_" + grupo] = {k: _r(100 * gm.intersection(u).area / area_t, 1)
                                            for k, u in uniones.items() if gm.intersects(u)}
            else:
                fila["elementos_que_cruzan"] = int(len(candidatos))
            filas.append(fila)
            afectados.append(f)

        if not es_area:
            clave = "elementos_que_cruzan"
        else:
            clave = "area_cubierta_km2" if _norm(order_by).startswith("area") else "porcentaje_cubierto"
        orden = sorted(range(len(filas)), key=lambda i: -filas[i][clave])
        top = max(1, min(int(top), 60))
        _al_mapa(lt, [afectados[i] for i in orden],
                 extra=[{clave: filas[i][clave]} for i in orden])
        return _dump({
            "filtro_overlay": clean_where(overlay_where),
            "elementos_overlay": len(capa_sup), "poligonos_evaluados": len(objetivos),
            "poligonos_afectados": len(filas), "campo_nombre": campo_t, "orden": clave,
            "resultado": [filas[i] for i in orden[:top]],
            "nota": f"se muestran {min(top, len(filas))} de {len(filas)}" if len(filas) > top else "",
        })
    except Exception as e:
        return _err(str(e), ayuda="Revisa los nombres de campo con get_layer_info.")


@tool
def nearest_features(target_url: str, lon: float, lat: float, n: int = 5,
                     max_km: float = 0, where: str = "1=1", name_field: str = "") -> str:
    """
    Elementos de una capa más cercanos a un punto (lon, lat), con su distancia en km.
    Úsala para: ¿qué parque está más cerca de Quito?, ¿qué hay a menos de 50 km de X?
      max_km : si es > 0, devuelve TODOS los que están a menos de esa distancia (hasta n=50)
      where  : filtro SQL opcional sobre la capa
    Si la pregunta nombra un lugar, obtén lon/lat antes con geocode_place.
    """
    try:
        layer = get_layer(target_url)
        campo = _campo_nombre(layer, name_field)
        pares = _preparar(_fetch(layer, where, [campo]))
        if not pares:
            return _dump({"total": 0, "resultado": []})
        lat, lon = float(lat), float(lon)
        pm = _a_metros(Point(lon, lat), lat)
        dist = [(f, _a_metros(g, lat).distance(pm) / 1000) for f, g in pares]
        dist.sort(key=lambda fd: fd[1])
        if float(max_km) > 0:
            dist = [fd for fd in dist if fd[1] <= float(max_km)]
            limite = 50
        else:
            limite = max(1, min(int(n), 20))
        sel = dist[:limite]
        _al_mapa(layer, [f for f, _ in sel], extra=[{"distancia_km": _r(d)} for _, d in sel])
        return _dump({
            "punto": {"lon": lon, "lat": lat}, "radio_km": float(max_km) or None,
            "total": len(dist) if float(max_km) > 0 else len(pares),
            "resultado": [{"nombre": _nombre(f["attributes"], campo, f"id {_oid(f['attributes'])}"),
                           "distancia_km": _r(d)} for f, d in sel[:MAX_FILAS_LLM * 2]],
        })
    except Exception as e:
        return _err(str(e))


@tool
def count_points_near(points_url: str, reference_url: str, distance_m: float = 500,
                      points_where: str = "1=1", reference_where: str = "1=1") -> str:
    """
    Cuántos puntos de points_url están a menos de distance_m metros de algún elemento
    de reference_url. Úsala para relacionar dos capas por cercanía: ¿cuántos reportes
    de seguridad hay a menos de 500 m de un accidente?
      points_where / reference_where : filtros SQL opcionales sobre cada capa
    """
    try:
        lp, lr = get_layer(points_url), get_layer(reference_url)
        puntos = _preparar(_fetch(lp, points_where))
        refs = _preparar(_fetch(lr, reference_where))
        if not puntos or not refs:
            return _dump({"total_puntos": len(puntos), "elementos_referencia": len(refs),
                          "puntos_cerca": 0, "porcentaje": 0.0})
        lat0 = _lat_centro(*[g for _, g in puntos[:200]])
        arbol = STRtree([_a_metros(g, lat0) for _, g in refs])
        pm = [_a_metros(g, lat0) for _, g in puntos]
        idx_pt, _ = arbol.query(pm, predicate="dwithin", distance=float(distance_m))
        cerca = sorted(set(idx_pt.tolist()))
        _al_mapa(lp, [puntos[i][0] for i in cerca])
        return _dump({
            "radio_m": float(distance_m), "filtro_puntos": clean_where(points_where),
            "filtro_referencia": clean_where(reference_where),
            "total_puntos": len(puntos), "elementos_referencia": len(refs),
            "puntos_cerca": len(cerca), "porcentaje": _r(100 * len(cerca) / len(puntos), 1),
        })
    except Exception as e:
        return _err(str(e), ayuda="Revisa los nombres de campo con get_layer_info.")


SPATIAL_TOOLS = [count_points_in_polygons, overlay_polygons, count_points_near,
                 nearest_features, locate_point, geocode_place]
