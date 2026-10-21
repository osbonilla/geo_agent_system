"""
scripts/survey123/generar_reportes.py
─────────────────────────────────────
Genera reportes ciudadanos SIMULADOS dentro del DMQ y los carga en la capa
del Survey123 "Reportes ciudadanos DMQ".

Los datos son sintéticos y así quedan marcados (campo origen = "simulado").
Tienen patrones conocidos a propósito, para que las preguntas de evaluación
tengan respuestas verificables:

  · cada parroquia tiene un perfil de problemas (urbanas: basura, ruido,
    transporte, seguridad; rurales: agua, vías, animales)
  · 3 parroquias "foco" concentran una categoría (se listan en resumen.md)
  · las inundaciones aumentan en temporada de lluvias (mar-may, oct-nov)
  · la satisfacción y el sentimiento del texto dependen de la gravedad y
    de si el problema se resolvió

La intención (queja, denuncia, solicitud, consulta, sugerencia, agradecimiento)
y el sentimiento reales NO se suben a la capa: quedan en verdad_sentimiento.csv,
para medir después la precisión de un clasificador (BERT) sin que el agente
pueda ver la respuesta.

Uso (dentro del contenedor de la API, que ya tiene las credenciales):

  # 1. Ver los campos de la capa de parroquias, para elegir el del nombre
  docker compose exec api python scripts/survey123/generar_reportes.py \\
      --parroquias-url https://services.arcgis.com/.../FeatureServer/0 --listar-campos

  # 2. Generar sin subir nada (revisa salida/ antes de cargar)
  docker compose exec api python scripts/survey123/generar_reportes.py \\
      --parroquias-url https://.../FeatureServer/0 --campo-nombre NOMBRE --n 1500

  # 3. Generar y cargar en la capa del survey
  docker compose exec api python scripts/survey123/generar_reportes.py \\
      --parroquias-url https://.../FeatureServer/0 --campo-nombre NOMBRE --n 1500 \\
      --destino-url https://.../survey123_xxx/FeatureServer/0 --subir

  # Borrar todo lo simulado (origen = 'simulado') de la capa destino
  docker compose exec api python scripts/survey123/generar_reportes.py \\
      --destino-url https://.../FeatureServer/0 --borrar-simulados
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

CATEGORIAS = ["vias", "alumbrado", "basura", "agua", "alcantarillado",
              "seguridad", "ruido", "transporte", "areas_verdes", "animales"]

# Peso base de cada categoría según el tipo de parroquia
PERFIL_URBANO = {"vias": 14, "alumbrado": 10, "basura": 16, "agua": 5, "alcantarillado": 8,
                 "seguridad": 15, "ruido": 12, "transporte": 12, "areas_verdes": 5, "animales": 3}
PERFIL_RURAL = {"vias": 20, "alumbrado": 10, "basura": 9, "agua": 18, "alcantarillado": 8,
                "seguridad": 7, "ruido": 3, "transporte": 9, "areas_verdes": 4, "animales": 12}

GRAVEDAD = {  # P(baja, media, alta) por categoría
    "vias": (0.25, 0.45, 0.30), "alumbrado": (0.35, 0.45, 0.20), "basura": (0.25, 0.45, 0.30),
    "agua": (0.15, 0.40, 0.45), "alcantarillado": (0.15, 0.35, 0.50), "seguridad": (0.10, 0.35, 0.55),
    "ruido": (0.40, 0.45, 0.15), "transporte": (0.30, 0.50, 0.20), "areas_verdes": (0.50, 0.40, 0.10),
    "animales": (0.40, 0.45, 0.15),
}
LLUVIAS = {3, 4, 5, 10, 11}            # meses de mayor lluvia en Quito
EDADES = (["18_29", "30_44", "45_59", "60_mas"], [0.28, 0.34, 0.24, 0.14])


# Los textos se generan por intención en textos.py (también lo usa ml/intencion/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textos import generar_texto, elegir_intencion, sentimiento_de  # noqa: E402


# ══════════════════════════════════════════════════════════
# GEOMETRÍA
# ══════════════════════════════════════════════════════════

def area_km2(poly) -> float:
    """Área aproximada en km² de un polígono en grados (latitud del DMQ ≈ 0°)."""
    lat = poly.centroid.y
    return poly.area * 111.32 * 111.32 * math.cos(math.radians(lat))


def punto_en(poly, centros, rnd: random.Random):
    """Punto dentro del polígono: 70 % alrededor de 'centros' (calles), 30 % uniforme."""
    from shapely.geometry import Point
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(500):
        if centros and rnd.random() < 0.7:
            cx, cy = rnd.choice(centros)
            p = Point(cx + rnd.gauss(0, 0.004), cy + rnd.gauss(0, 0.004))   # ~450 m
        else:
            p = Point(rnd.uniform(minx, maxx), rnd.uniform(miny, maxy))
        if poly.contains(p):
            return p
    return poly.representative_point()


def cargar_parroquias(args):
    """Devuelve [(nombre, polígono shapely en WGS84)]."""
    from shapely.geometry import shape
    if args.parroquias_geojson:
        data = json.load(open(args.parroquias_geojson, encoding="utf-8"))
        out = []
        for f in data["features"]:
            props = f.get("properties") or {}
            nombre = props.get(args.campo_nombre) if args.campo_nombre else None
            out.append((str(nombre or props.get("nombre") or f"P{len(out) + 1}"), shape(f["geometry"])))
        return out

    from tools.arcgis_tools import get_layer
    from arcgis.geometry import Geometry
    layer = get_layer(args.parroquias_url)
    fs = layer.query(where="1=1", out_fields=args.campo_nombre or "*", return_geometry=True, out_sr=4326)
    out = []
    for f in fs.features:
        if not f.geometry:
            continue
        # __geo_interface__ respeta los huecos (anillos interiores) del polígono
        geom = shape(Geometry(f.geometry).__geo_interface__).buffer(0)
        if geom.is_empty:
            continue
        out.append((str(f.attributes.get(args.campo_nombre, f"P{len(out) + 1}")), geom))
    return out


# ══════════════════════════════════════════════════════════
# GENERACIÓN
# ══════════════════════════════════════════════════════════

def generar(parroquias, n: int, seed: int, hasta: datetime):
    rnd = random.Random(seed)

    # Urbanas = las más pequeñas (en el DMQ las urbanas son mucho más chicas que las rurales)
    areas = sorted(area_km2(g) for _, g in parroquias)
    umbral = areas[len(areas) // 2] if len(areas) > 1 else float("inf")
    info = []
    for nombre, geom in parroquias:
        a = area_km2(geom)
        urbana = a <= umbral
        perfil = dict(PERFIL_URBANO if urbana else PERFIL_RURAL)
        for c in perfil:                                 # variación propia de cada parroquia
            perfil[c] *= rnd.uniform(0.6, 1.4)
        # Puntos "de calle" alrededor de los cuales se agrupan los reportes
        k = 6 if urbana else 3
        centros = []
        for _ in range(k):
            p = punto_en(geom, [], rnd)
            centros.append((p.x, p.y))
        peso = (2.2 if urbana else 1.4) * (1 + rnd.random())   # más reportes en lo urbano
        info.append({"nombre": nombre, "geom": geom, "urbana": urbana, "perfil": perfil,
                     "centros": centros, "peso": peso, "area_km2": round(a, 2)})

    # 3 focos conocidos: una parroquia urbana concentra una categoría
    urbanas = [p for p in info if p["urbana"]] or info
    focos = []
    for cat in ("basura", "seguridad", "alcantarillado"):
        p = rnd.choice(urbanas)
        p["perfil"][cat] *= 6
        p["peso"] *= 1.6
        focos.append((p["nombre"], cat))

    pesos = [p["peso"] for p in info]
    desde = hasta - timedelta(days=365)
    filas = []
    for i in range(n):
        p = rnd.choices(info, weights=pesos)[0]
        fecha = desde + timedelta(seconds=rnd.uniform(0, 365 * 86400))
        perfil = dict(p["perfil"])
        if fecha.month in LLUVIAS:
            perfil["alcantarillado"] *= 2.5
        cat = rnd.choices(list(perfil), weights=list(perfil.values()))[0]
        grav = rnd.choices(["baja", "media", "alta"], weights=GRAVEDAD[cat])[0]
        dias = (hasta - fecha).days
        p_res = min(0.60, 0.10 + dias / 500)              # lo viejo tiende a estar resuelto (~43 % en total)
        estado = rnd.choices(["resuelto", "en_proceso", "recibido"],
                             weights=[p_res, (1 - p_res) * 0.5, (1 - p_res) * 0.5])[0]

        # Sentimiento latente → satisfacción (1-5) y texto
        score = {"baja": 0.4, "media": 0.0, "alta": -0.6}[grav] + \
                {"resuelto": 0.9, "en_proceso": 0.0, "recibido": -0.5}[estado] + rnd.gauss(0, 0.6)
        # Umbrales calibrados por simulación: ~55 % negativo, ~32 % neutro, ~13 % positivo;
        # en un canal de reportes predominan las quejas.
        intencion, hartazgo = elegir_intencion(score, cat, rnd)
        sent = sentimiento_de(intencion, hartazgo)
        descripcion, plantilla = generar_texto(cat, intencion, rnd, hartazgo=hartazgo)
        satisf = max(1, min(5, round(3 + score * 1.3 + rnd.gauss(0, 0.5))))

        punto = punto_en(p["geom"], p["centros"], rnd)
        hora = rnd.choices(range(24), weights=[1, 1, 1, 1, 1, 2, 4, 6, 8, 8, 8, 8,
                                                7, 7, 7, 7, 8, 8, 9, 9, 8, 6, 4, 2])[0]
        fecha = fecha.replace(hour=hora, minute=rnd.randrange(60), second=0, microsecond=0)
        filas.append({
            "codigo": f"R-{i + 1:05d}",
            "lon": round(punto.x, 6), "lat": round(punto.y, 6),
            "categoria": cat, "gravedad": grav,
            "descripcion": descripcion,
            "satisfaccion": int(satisf),
            "recurrente": "si" if rnd.random() < (0.55 if sent == "negativo" else 0.25) else "no",
            "rango_edad": rnd.choices(*EDADES)[0],
            "fecha_reporte": fecha,
            "estado": estado,
            "origen": "simulado",
            # Solo para la verdad de evaluación (no se suben a la capa):
            "_parroquia": p["nombre"], "_sentimiento": sent,
            "_intencion": intencion, "_plantilla": plantilla,
        })
    return filas, info, focos


# ══════════════════════════════════════════════════════════
# SALIDAS
# ══════════════════════════════════════════════════════════

CAMPOS_CAPA = ["codigo", "categoria", "gravedad", "descripcion", "satisfaccion", "recurrente",
               "rango_edad", "fecha_reporte", "estado", "origen"]


def escribir(filas, info, focos, salida: str):
    os.makedirs(salida, exist_ok=True)
    # GeoJSON (para revisar en el mapa o cargar en AGOL a mano)
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [f["lon"], f["lat"]]},
         "properties": {k: (f[k].isoformat() if k == "fecha_reporte" else f[k]) for k in CAMPOS_CAPA}}
        for f in filas]}
    json.dump(gj, open(os.path.join(salida, "reportes_simulados.geojson"), "w", encoding="utf-8"),
              ensure_ascii=False)
    # CSV con lat/lon (Survey123 y AGOL lo importan directo)
    with open(os.path.join(salida, "reportes_simulados.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lat", "lon"] + CAMPOS_CAPA)
        for f in filas:
            w.writerow([f["lat"], f["lon"]] + [f[k].isoformat() if k == "fecha_reporte" else f[k]
                                               for k in CAMPOS_CAPA])
    # Verdad para evaluar: sentimiento y parroquia real de cada reporte
    with open(os.path.join(salida, "verdad_sentimiento.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["codigo", "parroquia", "categoria", "intencion", "sentimiento", "plantilla", "descripcion"])
        for f in filas:
            w.writerow([f["codigo"], f["_parroquia"], f["categoria"], f["_intencion"], f["_sentimiento"],
                        f["_plantilla"], f["descripcion"]])

    # Resumen con las respuestas conocidas
    por_parr = Counter(f["_parroquia"] for f in filas)
    por_cat = Counter(f["categoria"] for f in filas)
    por_sent = Counter(f["_sentimiento"] for f in filas)
    por_int = Counter(f["_intencion"] for f in filas)
    cat_parr = defaultdict(Counter)
    for f in filas:
        cat_parr[f["categoria"]][f["_parroquia"]] += 1
    L = ["# Reportes ciudadanos simulados · resumen", "",
         f"- Reportes: **{len(filas)}** · parroquias: **{len(info)}** "
         f"({sum(p['urbana'] for p in info)} urbanas, {sum(not p['urbana'] for p in info)} rurales)",
         f"- Sentimiento: " + ", ".join(f"{k} {v} ({v / len(filas):.0%})" for k, v in por_sent.most_common()),
         f"- Intención: " + ", ".join(f"{k} {v} ({v / len(filas):.0%})" for k, v in por_int.most_common()),
         "", "## Focos conocidos (patrón introducido a propósito)", "",
         "| Parroquia | Categoría concentrada | Reportes de esa categoría | Puesto en esa categoría |",
         "|---|---|---|---|"]
    for parr, cat in focos:
        ranking = [p for p, _ in cat_parr[cat].most_common()]
        L.append(f"| {parr} | {cat} | {cat_parr[cat][parr]} | {ranking.index(parr) + 1 if parr in ranking else '—'} |")
    L += ["", "## Reportes por categoría", "", "| Categoría | Reportes | Parroquia con más |", "|---|---|---|"]
    for c, v in por_cat.most_common():
        top, nt = cat_parr[c].most_common(1)[0]
        L.append(f"| {c} | {v} | {top} ({nt}) |")
    L += ["", "## Parroquias con más reportes", "", "| Parroquia | Reportes |", "|---|---|"]
    for p, v in por_parr.most_common(10):
        L.append(f"| {p} | {v} |")
    open(os.path.join(salida, "resumen.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def a_features(filas):
    feats = []
    for f in filas:
        attrs = {k: f[k] for k in CAMPOS_CAPA}
        attrs["fecha_reporte"] = int(f["fecha_reporte"].astimezone(timezone.utc).timestamp() * 1000)
        feats.append({"geometry": {"x": f["lon"], "y": f["lat"], "spatialReference": {"wkid": 4326}},
                      "attributes": attrs})
    return feats


def subir(layer, filas, lote: int = 250) -> int:
    campos = {fld["name"].lower(): fld["name"] for fld in layer.properties["fields"]}
    faltan = [c for c in CAMPOS_CAPA if c.lower() not in campos]
    if faltan:
        raise SystemExit(f"La capa destino no tiene los campos {faltan}. ¿Publicaste el formulario "
                         f"reportes_ciudadanos_dmq.xlsx con Survey123 Connect? Campos de la capa: "
                         f"{sorted(campos.values())}")
    feats = a_features(filas)
    for ft in feats:                                   # respeta mayúsculas reales de la capa
        ft["attributes"] = {campos[k.lower()]: v for k, v in ft["attributes"].items()}
    ok = 0
    for i in range(0, len(feats), lote):
        r = layer.edit_features(adds=feats[i:i + lote])
        res = r.get("addResults", [])
        bien = sum(1 for x in res if x.get("success"))
        ok += bien
        print(f"  lote {i // lote + 1}: {bien}/{len(res)} agregados")
        errores = [x.get("error") for x in res if not x.get("success")]
        if errores:
            print(f"    primer error: {errores[0]}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Reportes ciudadanos simulados para el DMQ")
    ap.add_argument("--parroquias-url", help="URL del Feature Layer de parroquias del DMQ")
    ap.add_argument("--parroquias-geojson", help="Alternativa: parroquias en un GeoJSON (WGS84)")
    ap.add_argument("--campo-nombre", help="Campo con el nombre de la parroquia")
    ap.add_argument("--listar-campos", action="store_true", help="Muestra los campos de la capa de parroquias y sale")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42, help="Misma semilla = mismos datos (reproducible)")
    ap.add_argument("--hasta", default=None, help="Fecha final AAAA-MM-DD (por defecto, hoy). Se cubren 365 días")
    ap.add_argument("--salida", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "salida"))
    ap.add_argument("--destino-url", help="URL de la capa del survey (…/FeatureServer/0)")
    ap.add_argument("--subir", action="store_true", help="Cargar en la capa destino (sin esto, solo genera archivos)")
    ap.add_argument("--borrar-simulados", action="store_true", help="Borra de la capa destino lo que tenga origen = 'simulado'")
    a = ap.parse_args()

    if a.borrar_simulados:
        from tools.arcgis_tools import get_layer
        layer = get_layer(a.destino_url)
        r = layer.delete_features(where="origen = 'simulado'")
        print(f"Borrados: {sum(1 for x in r.get('deleteResults', []) if x.get('success'))}")
        return

    if a.listar_campos:
        from tools.arcgis_tools import get_layer
        urls = [(t, u) for t, u in (("Parroquias", a.parroquias_url), ("Destino", a.destino_url)) if u]
        if not urls:
            ap.error("--listar-campos necesita --parroquias-url y/o --destino-url")
        for titulo, url in urls:
            layer = get_layer(url)
            print(f"\n[{titulo}] {layer.properties.get('name')} · {layer.query(return_count_only=True)} registros")
            for fld in layer.properties["fields"]:
                print(f"  {fld['name']:<24} {fld.get('alias', ''):<30} {fld['type']}")
        return

    if not (a.parroquias_url or a.parroquias_geojson):
        ap.error("Indica --parroquias-url o --parroquias-geojson")

    hasta = (datetime.strptime(a.hasta, "%Y-%m-%d") if a.hasta else datetime.now()) \
        .replace(tzinfo=timezone(timedelta(hours=-5)))       # hora de Ecuador
    parroquias = cargar_parroquias(a)
    print(f"Parroquias: {len(parroquias)}")
    filas, info, focos = generar(parroquias, a.n, a.seed, hasta)
    escribir(filas, info, focos, a.salida)
    print(f"Generados {len(filas)} reportes → {a.salida}")
    print("Focos: " + ", ".join(f"{p} ({c})" for p, c in focos))

    if a.subir:
        if not a.destino_url:
            ap.error("--subir necesita --destino-url")
        from tools.arcgis_tools import get_layer
        print(f"Cargando en {a.destino_url} …")
        ok = subir(get_layer(a.destino_url), filas)
        print(f"Listo: {ok}/{len(filas)} reportes cargados")
    else:
        print("Modo prueba: no se subió nada. Revisa salida/ y agrega --destino-url … --subir")


if __name__ == "__main__":
    main()