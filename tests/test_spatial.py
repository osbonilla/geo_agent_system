"""
tests/test_spatial.py
─────────────────────
Tests offline del SpatialAgent y sus herramientas (sin ArcGIS, sin LLM, sin red).

    pytest tests/test_spatial.py -v

Escenario: dos parroquias cuadradas de ~1,1 km de lado, una junto a la otra,
reportes ciudadanos (puntos) y una zona de amenaza que cubre media parroquia.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fakes import FakeGeoLayer, ScriptedToolModel, square, tool_call, text  # noqa: E402

PARROQUIAS = "https://x/parroquias/FeatureServer/2"
REPORTES = "https://x/reportes/FeatureServer/0"
AMENAZA = "https://x/amenaza/FeatureServer/0"
ACCIDENTES = "https://x/accidentes/FeatureServer/0"


def pt(x, y):
    return {"x": x, "y": y}


@pytest.fixture
def capas(monkeypatch):
    parroquias = FakeGeoLayer(PARROQUIAS, "Parroquias_DMQ", "esriGeometryPolygon",
                              [("dpa_parroq", "dpa_parroq", "esriFieldTypeString"),
                               ("dpa_despar", "dpa_despar", "esriFieldTypeString")],
                              [({"dpa_parroq": "170150", "dpa_despar": "LA CONCEPCIÓN"},
                                square(-78.50, -0.20, -78.49, -0.19)),
                               ({"dpa_parroq": "170151", "dpa_despar": "QUITUMBE"},
                                square(-78.49, -0.20, -78.48, -0.19))])
    reportes = FakeGeoLayer(REPORTES, "Reportes", "esriGeometryPoint",
                            [("categoria", "Tipo de problema", "esriFieldTypeString"),
                             ("intencion_bert", "Intención (BERT)", "esriFieldTypeString")],
                            [({"categoria": "seguridad", "intencion_bert": "denuncia"}, pt(-78.495, -0.195)),
                             ({"categoria": "seguridad", "intencion_bert": "denuncia"}, pt(-78.496, -0.196)),
                             ({"categoria": "basura", "intencion_bert": "queja"}, pt(-78.497, -0.194)),
                             ({"categoria": "basura", "intencion_bert": "queja"}, pt(-78.485, -0.195)),
                             ({"categoria": "vias", "intencion_bert": "denuncia"}, pt(-78.40, -0.10))])  # fuera
    amenaza = FakeGeoLayer(AMENAZA, "Amenaza", "esriGeometryPolygon",
                           [("tipo", "Tipo de amenaza volcánica", "esriFieldTypeString"),
                            ("volcan", "Nombre del volcán", "esriFieldTypeString")],
                           [({"tipo": "Lahares", "volcan": "Cotopaxi"}, square(-78.50, -0.20, -78.495, -0.19))])
    accidentes = FakeGeoLayer(ACCIDENTES, "Accidentes", "esriGeometryPoint",
                              [("tipo", "Tipo", "esriFieldTypeString")],
                              [({"tipo": "choque"}, pt(-78.4952, -0.1952))])
    por_url = {c.url: c for c in (parroquias, reportes, amenaza, accidentes)}

    import tools.spatial_tools as sp
    import tools.arcgis_tools as at
    monkeypatch.setattr(sp, "get_layer", lambda url: por_url[url])
    monkeypatch.setattr(at, "get_layer", lambda url: por_url[url])
    sp.clear_cache()
    return por_url


def call(tool, **args):
    return json.loads(tool.invoke(args))


# ── Geometrías ───────────────────────────────────────────

def test_esri_polygon_with_hole():
    from tools.spatial_tools import esri_to_shapely
    exterior = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]      # horario
    hueco = [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]             # antihorario
    g = esri_to_shapely({"rings": [exterior, hueco]})
    assert g.geom_type == "Polygon" and g.area == pytest.approx(96)


def test_local_projection_area_is_close_to_real():
    """Un cuadrado de 0,01° en el ecuador mide ~1,113 km × 1,106 km."""
    from tools.spatial_tools import esri_to_shapely, _a_metros
    g = _a_metros(esri_to_shapely(square(-78.50, -0.20, -78.49, -0.19)), -0.195)
    assert g.area / 1e6 == pytest.approx(1.2308, rel=0.005)


# ── Herramientas ─────────────────────────────────────────

def test_count_points_in_polygons_ranking(capas):
    from tools import result_store
    from tools.spatial_tools import count_points_in_polygons
    result_store.reset()
    out = call(count_points_in_polygons, points_url=REPORTES, polygons_url=PARROQUIAS)
    assert out["campo_nombre"] == "dpa_despar"                    # detectado solo
    assert out["ranking"][0] == {"nombre": "LA CONCEPCIÓN", "conteo": 3, "porcentaje": 75.0}
    assert out["puntos_fuera"] == 1 and out["total_puntos"] == 5
    assert result_store.items()[0]["attributes"]["conteo"] == 3   # el mapa recibe los polígonos


def test_count_points_with_filter_name_and_breakdown(capas):
    from tools.spatial_tools import count_points_in_polygons
    out = call(count_points_in_polygons, points_url=REPORTES, polygons_url=PARROQUIAS,
               points_where="intencion_bert = 'denuncia'", polygon_name="la concepcion",
               group_field="Tipo de problema")                     # alias, sin tildes
    assert out["ranking"] == [{"nombre": "LA CONCEPCIÓN", "conteo": 2, "porcentaje": 100.0,
                               "por_categoria": {"seguridad": 2}}]


def test_unknown_polygon_name_lists_valid_names(capas):
    from tools.spatial_tools import count_points_in_polygons
    out = call(count_points_in_polygons, points_url=REPORTES, polygons_url=PARROQUIAS,
               polygon_name="Cumbayá")
    assert out["status"] == "error" and "QUITUMBE" in out["nombres_validos"]


def test_overlay_polygons_percentage_and_groups(capas):
    from tools.spatial_tools import overlay_polygons
    out = call(overlay_polygons, target_url=PARROQUIAS, overlay_url=AMENAZA, group_field="tipo")
    assert out["poligonos_afectados"] == 1
    fila = out["resultado"][0]
    assert fila["nombre"] == "LA CONCEPCIÓN"
    assert fila["porcentaje_cubierto"] == pytest.approx(50, abs=0.5)
    assert fila["por_tipo"] == {"Lahares": pytest.approx(50, abs=0.5)}


def test_overlay_by_name_and_empty_filter(capas):
    from tools.spatial_tools import overlay_polygons
    assert call(overlay_polygons, target_url=PARROQUIAS, overlay_url=AMENAZA,
                overlay_name="cotopaxi")["poligonos_afectados"] == 1
    out = call(overlay_polygons, target_url=PARROQUIAS, overlay_url=AMENAZA, overlay_name="Pichincha")
    assert out["status"] == "error" and out["nombres_validos"] == ["Cotopaxi"]


def test_count_points_near_other_layer(capas):
    from tools.spatial_tools import count_points_near
    out = call(count_points_near, points_url=REPORTES, reference_url=ACCIDENTES, distance_m=300)
    assert out["puntos_cerca"] == 3 and out["total_puntos"] == 5   # los 3 de La Concepción
    assert call(count_points_near, points_url=REPORTES, reference_url=ACCIDENTES,
                distance_m=300, points_where="intencion_bert = 'queja'")["puntos_cerca"] == 1


def test_nearest_and_within_radius(capas):
    from tools.spatial_tools import nearest_features
    out = call(nearest_features, target_url=PARROQUIAS, lon=-78.47, lat=-0.195, n=1)
    assert out["resultado"][0]["nombre"] == "QUITUMBE"
    assert out["resultado"][0]["distancia_km"] == pytest.approx(1.11, abs=0.02)
    out = call(nearest_features, target_url=PARROQUIAS, lon=-78.47, lat=-0.195, max_km=1.5)
    assert [r["nombre"] for r in out["resultado"]] == ["QUITUMBE"]


def test_locate_point_inside_and_outside(capas):
    from tools.spatial_tools import locate_point
    assert call(locate_point, polygons_url=PARROQUIAS, lon=-78.485, lat=-0.195)["dentro_de"] == ["QUITUMBE"]
    out = call(locate_point, polygons_url=PARROQUIAS, lon=-78.46, lat=-0.195)
    assert out["dentro_de"] == [] and out["mas_cercano"] == "QUITUMBE"


def test_layers_are_downloaded_once(capas):
    from tools.spatial_tools import count_points_in_polygons
    for _ in range(3):
        call(count_points_in_polygons, points_url=REPORTES, polygons_url=PARROQUIAS)
    assert capas[REPORTES].calls == 1 and capas[PARROQUIAS].calls == 1


def test_llm_never_sees_geometries(capas):
    from tools.spatial_tools import count_points_in_polygons, overlay_polygons
    for raw in (count_points_in_polygons.invoke({"points_url": REPORTES, "polygons_url": PARROQUIAS}),
                overlay_polygons.invoke({"target_url": PARROQUIAS, "overlay_url": AMENAZA})):
        assert "rings" not in raw and "-78.49" not in raw


# ── Agente y Planner ─────────────────────────────────────

@pytest.fixture
def model(monkeypatch):
    from llm.metrics import COUNTER
    import llm.router as router
    import llm.structured as structured
    import agents.base as base
    holder = {}

    def install(script):
        holder["m"] = ScriptedToolModel(script=list(script), callbacks=[COUNTER])
        return holder["m"]

    for mod in (router, structured, base):
        monkeypatch.setattr(mod, "get_llm", lambda *a, **k: holder["m"])
    monkeypatch.setattr(structured, "get_llm_info", lambda: {"provider": "ollama"})
    monkeypatch.setattr(base, "grounding_context", lambda *a, **k: "")
    return install


def test_spatial_agent_tool_calling(capas, model):
    from agents.spatial_agent import run_spatial_agent
    model([tool_call("count_points_in_polygons", {"points_url": REPORTES, "polygons_url": PARROQUIAS,
                                                  "points_where": "intencion_bert = 'denuncia'"}),
           text("La parroquia con más denuncias es LA CONCEPCIÓN (2).")])
    r = run_spatial_agent("¿Qué parroquia tiene más denuncias?")
    assert r["success"] and "LA CONCEPCIÓN" in r["output"]
    assert [s["tool"] for s in r["steps"]] == ["count_points_in_polygons"]


def test_planner_routes_cross_layer_question_to_spatial_agent(capas, model):
    from agents.dispatch import dispatch
    m = model([
        text(json.dumps({"tareas": [{"agente": "spatial", "tarea":
             f"Contar denuncias ({REPORTES}) por parroquia ({PARROQUIAS})"}]})),
        tool_call("count_points_in_polygons", {"points_url": REPORTES, "polygons_url": PARROQUIAS,
                                               "points_where": "intencion_bert = 'denuncia'"}),
        text("LA CONCEPCIÓN, con 2 denuncias."),
    ])
    r = dispatch("¿Qué parroquia tiene más denuncias?", persist=False)
    assert r["plan"][0]["agente"] == "SpatialAgent"                # 'spatial' normalizado
    assert r["answer"].startswith("LA CONCEPCIÓN")
    assert len(r["geojson"]["features"]) == 1                     # el polígono va al mapa
    assert "SpatialAgent" in str(m.seen[0])                        # el Planner conoce al agente


def test_spatial_prefix_routes_directly():
    from agents.dispatch import route
    assert route("spatial: accidentes por parroquia", None) == ("SpatialAgent", "accidentes por parroquia")


def test_eval_reference_can_use_spatial_tools(capas):
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from run_eval import reference_values
    v = reference_values({"herramienta": "count_points_in_polygons",
                          "args": {"points_url": REPORTES, "polygons_url": PARROQUIAS},
                          "clave": "ranking.0.nombre"})
    assert v == ["LA CONCEPCIÓN"]


# ── Regresiones ──────────────────────────────────────────

@pytest.mark.parametrize("entrada,esperado", [
    ("gravedad = 'alta'", "gravedad = 'alta'"),              # antes se perdía la comilla final
    ("'categoria = basura'", "categoria = basura"),          # filtro envuelto entero en comillas
    ('"intencion_bert = \'queja\'"', "intencion_bert = 'queja'"),
    ("", "1=1"), (None, "1=1"),
])
def test_where_keeps_its_quotes(entrada, esperado):
    from tools.arcgis_tools import clean_where
    assert clean_where(entrada) == esperado


def test_overlay_order_by_area_vs_percentage(capas, monkeypatch):
    """Una zona chica casi cubierta gana por porcentaje; una grande, por área."""
    from tools.spatial_tools import overlay_polygons, clear_cache
    grande = square(-78.60, -0.30, -78.50, -0.20)                # 100 veces más grande
    capas[PARROQUIAS].features.append(({"OBJECTID": 3, "dpa_despar": "LLOA"}, grande))
    capas[AMENAZA].features.append(({"OBJECTID": 2, "tipo": "Flujos", "volcan": "Pichincha"},
                                    square(-78.60, -0.30, -78.55, -0.20)))   # media Lloa
    clear_cache()
    por_pct = call(overlay_polygons, target_url=PARROQUIAS, overlay_url=AMENAZA)
    por_area = call(overlay_polygons, target_url=PARROQUIAS, overlay_url=AMENAZA, order_by="area")
    assert por_pct["resultado"][0]["porcentaje_cubierto"] == por_area["resultado"][0]["porcentaje_cubierto"] == pytest.approx(50, abs=0.5)
    assert por_area["resultado"][0]["nombre"] == "LLOA"
    assert por_area["resultado"][0]["porcentaje_del_overlay"] == pytest.approx(99, abs=0.5)   # el 1 % restante cae en La Concepción


def test_only_needed_fields_are_downloaded(capas):
    """La capa de accidentes real tiene ~200 campos: solo se piden los que usa el cálculo."""
    from tools.spatial_tools import count_points_in_polygons
    call(count_points_in_polygons, points_url=REPORTES, polygons_url=PARROQUIAS, group_field="categoria")
    assert capas[REPORTES].last_kw["out_fields"] == "OBJECTID,categoria"
    assert capas[PARROQUIAS].last_kw["out_fields"] == "OBJECTID,dpa_despar"


def test_layer_info_is_compact(capas):
    from tools.arcgis_tools import get_layer_info
    out = call(get_layer_info, layer_url=REPORTES)
    assert "categoria (Tipo de problema): texto" in out["campos"]
    assert "intencion_bert: texto" in out["campos"]          # alias redundante: se omite
    assert "OBJECTID: id" in out["campos"]


def test_category_values_are_indexed_for_grounding():
    """El modelo debe ver que 'denuncias' es un valor de intencion_bert."""
    from fakes import FakeLayer
    from tools.vector_tools import category_values, format_values, build_documents
    vals = category_values(FakeLayer())
    assert vals == {"field_7": ["Eucalipto", "Arupo", "Cholán", "Pumamaqui"]}
    doc = build_documents({"id": "w", "titulo": "T", "snippet": "", "descripcion": "", "propietario": "",
                           "tags": [], "basemap": "", "total_capas": 1,
                           "capas": [{"id": "c", "titulo": "Árboles", "tipo": "FL", "url": "u",
                                      "campos": [], "valores": vals}]})[1]["text"]
    assert "Valores: field_7 = Eucalipto|Arupo|Cholán|Pumamaqui" in doc
    assert format_values({"a": ["x"] * 400}, max_chars=50) == ""


# ── Tolerancia a errores frecuentes de los SLM (vistos en las trazas) ──

def test_polygon_name_field_passed_as_group_field_is_tolerated(capas):
    from tools.spatial_tools import count_points_in_polygons
    out = call(count_points_in_polygons, points_url=REPORTES, polygons_url=PARROQUIAS, group_field="dpa_despar")
    assert out["ranking"][0]["nombre"] == "LA CONCEPCIÓN" and "por_dpa_despar" not in out["ranking"][0]


def test_polygon_layer_as_points_suggests_overlay(capas):
    from tools.spatial_tools import count_points_in_polygons
    out = call(count_points_in_polygons, points_url=AMENAZA, polygons_url=PARROQUIAS,
               points_where="volcan = 'Cotopaxi'")
    assert out["status"] == "error" and "overlay_polygons" in out["mensaje"] and AMENAZA in out["mensaje"]


def test_empty_filter_warns_instead_of_concluding(capas):
    from tools.spatial_tools import count_points_in_polygons
    out = call(count_points_in_polygons, points_url=REPORTES, polygons_url=PARROQUIAS,
               points_where="intencion_bert = 'FALLECIDOS'")
    assert out["total_puntos"] == 0 and "get_layer_info" in out["aviso"]


def test_unknown_field_error_lists_similar_fields_first(capas):
    from tools.spatial_tools import count_points_in_polygons
    out = call(count_points_in_polygons, points_url=REPORTES, polygons_url=PARROQUIAS, group_field="intencion")
    assert out["mensaje"].split("Campos válidos: ")[1].startswith("intencion_bert")


def test_empty_answer_without_tools_is_retried(capas, model):
    from agents.spatial_agent import run_spatial_agent
    model([text(""), tool_call("count_points_in_polygons", {"points_url": REPORTES, "polygons_url": PARROQUIAS}),
           text("LA CONCEPCIÓN")])
    r = run_spatial_agent("¿qué parroquia tiene más reportes?")
    assert r["output"] == "LA CONCEPCIÓN" and r["steps_count"] == 1


def test_empty_filter_shows_real_values_of_that_field(capas):
    from tools.spatial_tools import count_points_in_polygons
    out = call(count_points_in_polygons, points_url=REPORTES, polygons_url=PARROQUIAS,
               points_where="categoria = 'Seguridad ciudadana'")
    assert out["valores_de_categoria"] == ["seguridad", "basura", "vias"]


def test_tool_call_written_as_text_is_retried(capas, model):
    from agents.spatial_agent import run_spatial_agent
    model([text(json.dumps({"points_url": REPORTES, "polygons_url": PARROQUIAS})),     # escribió la llamada
           tool_call("count_points_in_polygons", {"points_url": REPORTES, "polygons_url": PARROQUIAS}),
           text("LA CONCEPCIÓN")])
    r = run_spatial_agent("¿qué parroquia tiene más reportes?")
    assert r["output"] == "LA CONCEPCIÓN" and r["steps_count"] == 1


def test_empty_final_answer_after_tools_is_written_from_tool_output(capas, model):
    from agents.spatial_agent import run_spatial_agent
    m = model([tool_call("count_points_in_polygons", {"points_url": REPORTES, "polygons_url": PARROQUIAS}),
               text(""),                                       # respuesta final vacía
               text("La parroquia con más reportes es LA CONCEPCIÓN (3).")])
    r = run_spatial_agent("¿qué parroquia tiene más reportes?")
    assert "LA CONCEPCIÓN" in r["output"] and "LA CONCEPCIÓN" in str(m.seen[-1])


def test_overlay_tolerates_target_name_field_as_group_field(capas):
    from tools.spatial_tools import overlay_polygons
    out = call(overlay_polygons, target_url=PARROQUIAS, overlay_url=AMENAZA, group_field="dpa_despar")
    assert out["resultado"][0]["nombre"] == "LA CONCEPCIÓN"


def test_written_answer_uses_last_useful_tool_not_auxiliary(capas, model):
    from agents.spatial_agent import run_spatial_agent
    m = model([tool_call("count_points_in_polygons", {"points_url": REPORTES, "polygons_url": PARROQUIAS}, 0),
               tool_call("get_layer_info", {"layer_url": PARROQUIAS}, 1),
               text(""),
               text("LA CONCEPCIÓN")])
    r = run_spatial_agent("¿qué parroquia tiene más reportes?")
    assert r["output"] == "LA CONCEPCIÓN"
    assert "count_points_in_polygons" in str(m.seen[-1]) and "ranking" in str(m.seen[-1])
