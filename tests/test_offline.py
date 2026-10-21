"""
tests/test_offline.py
─────────────────────
Tests que corren SIN servicios: ni ArcGIS, ni LLM, ni PostgreSQL, ni red.

    pip install pytest
    pytest tests/test_offline.py -v

Verifican la mecánica del sistema (ruteo, tool calling, salida estructurada,
canal lateral de geometrías, métricas, evaluación). La CALIDAD de un SLM
concreto se mide con eval/run_eval.py contra datos reales.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CHROMA_PATH", os.path.join(ROOT, ".chroma_test"))

from fakes import ScriptedToolModel, FakeLayer, tool_call, text  # noqa: E402


@pytest.fixture
def layer(monkeypatch):
    fl = FakeLayer()
    import tools.arcgis_tools as at
    import tools.analytics_tools as an
    monkeypatch.setattr(at, "get_layer", lambda url: fl)
    monkeypatch.setattr(an, "get_layer", lambda url: fl)
    return fl


@pytest.fixture
def model(monkeypatch):
    """Instala un modelo guionado como LLM activo de todo el sistema."""
    from llm.metrics import COUNTER
    import llm.router as router
    import llm.structured as structured
    import agents.base as base
    import agents.rag_agent as rag

    holder = {}

    def install(script):
        m = ScriptedToolModel(script=list(script), callbacks=[COUNTER])
        holder["m"] = m
        return m

    get = lambda *a, **k: holder["m"]  # noqa: E731
    for mod in (router, structured, base, rag):
        monkeypatch.setattr(mod, "get_llm", get)
    monkeypatch.setattr(structured, "get_llm_info", lambda: {"provider": "ollama"})
    # sin índice vectorial en los tests
    monkeypatch.setattr(base, "grounding_context", lambda *a, **k: "")
    return install


# ── Herramientas ─────────────────────────────────────────

def test_field_alias_is_resolved(layer):
    from tools.analytics_tools import field_statistics
    out = json.loads(field_statistics.invoke({"layer_url": "x", "field_name": "Altura total (m)"}))
    assert out["campo"] == "_3_altura_total"
    assert out["maximo"] == 32.5


def test_unknown_field_lists_valid_fields(layer):
    from tools.analytics_tools import field_statistics
    out = json.loads(field_statistics.invoke({"layer_url": "x", "field_name": "altura"}))
    assert out["status"] == "error"
    assert "_3_altura_total" in out["mensaje"] and "Especie" in out["mensaje"]


def test_query_output_is_compact_and_geometry_goes_to_store(layer):
    from tools import result_store
    from tools.arcgis_tools import query_features, MAX_ROWS_TO_LLM
    result_store.reset()
    raw = query_features.invoke({"layer_url": "x", "where": "field_7 = 'Eucalipto'", "max_records": 50})
    out = json.loads(raw)
    assert out["total_que_cumplen"] == 8
    assert len(out["muestra"]) <= MAX_ROWS_TO_LLM
    assert "geometria" not in raw and '"x"' not in raw       # el LLM no ve geometrías
    assert len(result_store.items()) == 8                     # el mapa sí


def test_spatial_query_uses_distance(layer):
    from tools.arcgis_tools import query_features_near
    out = json.loads(query_features_near.invoke(
        {"layer_url": "x", "lon": -78.4678, "lat": -0.1807, "distance_m": 300}))
    assert out["total_en_radio"] < 20
    assert layer.calls[-1]["units"] == "esriSRUnit_Meter"


def test_normalize_layer_url():
    from tools.arcgis_tools import normalize_layer_url
    assert normalize_layer_url(" https://a/FeatureServer/ ").endswith("/FeatureServer/0")
    assert normalize_layer_url("https://a/FeatureServer/2/query").endswith("/FeatureServer/2")


# ── Agentes con tool calling ─────────────────────────────

def test_analytics_agent_tool_calling(layer, model):
    from agents.analytics_agent import run_analytics_agent
    model([tool_call("field_statistics", {"layer_url": "x", "field_name": "Altura total (m)"}),
           text("La altura máxima es 32.5 m.")])
    r = run_analytics_agent("¿altura máxima?")
    assert r["success"] and "32.5" in r["output"]
    assert [s["tool"] for s in r["steps"]] == ["field_statistics"]


def test_agent_recovers_from_wrong_field(layer, model):
    """El error con la lista de campos permite que el modelo se corrija."""
    from agents.analytics_agent import run_analytics_agent
    m = model([tool_call("count_by_category", {"layer_url": "x", "field_name": "especie_arbol"}, 0),
               tool_call("count_by_category", {"layer_url": "x", "field_name": "field_7"}, 1),
               text("La especie más común es Eucalipto con 8 árboles.")])
    r = run_analytics_agent("¿especie más común?")
    assert r["steps_count"] == 2 and "Campos válidos" in r["steps"][0]["output"]
    assert "Eucalipto" in r["output"]
    assert "Campos válidos" in str(m.seen[1])     # el error volvió al modelo


# ── Planner con salida estructurada ──────────────────────

def test_planner_routes_and_passes_context(layer, model):
    from agents.planner import run_planner
    plan = {"tareas": [{"agente": "GeoAgent", "tarea": "buscar la capa de arbolado"},
                       {"agente": "AnalyticsAgent", "tarea": "especie más común"}]}
    m = model([
        text("```json\n" + json.dumps(plan) + "\n```"),          # planner (respaldo de texto)
        tool_call("count_features", {"layer_url": FakeLayer.url}),  # GeoAgent
        text(f"Capa encontrada: {FakeLayer.url}"),
        tool_call("count_by_category", {"layer_url": FakeLayer.url, "field_name": "Especie"}),
        text("Eucalipto (8)."),                                    # AnalyticsAgent
        text(json.dumps({"decision": "finalizar", "respuesta_final": "La más común es Eucalipto (8)."})),
    ])
    r = run_planner("¿cuál es la especie más común?")
    assert r["agentes"] == ["GeoAgent", "AnalyticsAgent"]
    assert r["final_answer"].startswith("La más común es Eucalipto")
    analytics_input = str(m.seen[3])
    assert "CONTEXTO DE PASOS ANTERIORES" in analytics_input and FakeLayer.url in analytics_input


def test_single_task_plan_skips_replan(layer, model):
    from agents.dispatch import dispatch
    model([text(json.dumps({"tareas": [{"agente": "AnalyticsAgent", "tarea": "altura máxima"}]})),
           tool_call("field_statistics", {"layer_url": "x", "field_name": "_3_altura_total"}),
           text("32.5 m")])
    r = dispatch("¿altura máxima?", persist=False)
    assert r["answer"] == "32.5 m"
    assert r["metrics"]["llm_calls"] == 3                   # planner + 2 del agente, sin replan
    assert r["metrics"]["prompt_tokens"] == 300


def test_dispatch_returns_geojson_for_map(layer, model):
    from agents.dispatch import dispatch
    model([tool_call("top_n_features", {"layer_url": "x", "field_name": "_3_altura_total", "n": 3}),
           text("Los 3 más altos...")])
    r = dispatch("analytics: top 3 más altos", persist=False)
    assert r["agent_used"] == "AnalyticsAgent"
    assert len(r["geojson"]["features"]) == 3
    assert r["geojson"]["features"][0]["properties"]["_3_altura_total"] == 32.5


def test_structured_output_preferred_when_supported(monkeypatch):
    """Si el proveedor soporta json_schema, no se parsea texto."""
    from pydantic import BaseModel
    import llm.structured as st

    class Out(BaseModel):
        a: int

    class M:
        def with_structured_output(self, schema, **kw):
            from langchain_core.runnables import RunnableLambda
            assert kw.get("method") == "json_schema"
            return RunnableLambda(lambda _: {"raw": None, "parsed": schema(a=1), "parsing_error": None})

    monkeypatch.setattr(st, "get_llm", lambda: M())
    monkeypatch.setattr(st, "get_llm_info", lambda: {"provider": "ollama"})
    from langchain_core.prompts import ChatPromptTemplate
    assert st.invoke_structured(ChatPromptTemplate.from_messages([("human", "x")]), {}, Out).a == 1


# ── Seguridad y evaluación ───────────────────────────────

def test_sql_tool_rejects_multiple_statements():
    from tools.postgis_tools import run_sql_spatial
    out = json.loads(run_sql_spatial.invoke({"sql_query": "SELECT 1; DROP TABLE sessions"}))
    assert out["status"] == "error"


@pytest.mark.parametrize("answer,expected,ok", [
    ("La altura máxima es 32,5 metros", 32.5, True),
    ("Hay 1.250 árboles", 1250, True),
    ("Promedio: 14.27", 14.271, True),
    ("La altura es 30 m", 32.5, False),
    ("La especie más común es el CHOLÁN", "Cholán", True),
])
def test_eval_answer_matching(answer, expected, ok):
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from run_eval import contains_value
    assert contains_value(answer, expected) is ok


# ── Índice vectorial: cambio de modelo de embeddings ─────

def test_embedding_change_is_detected_and_index_regenerates(tmp_path, monkeypatch):
    import hashlib
    import llm.embeddings as E
    import tools.vector_tools as vt

    monkeypatch.setattr(vt.settings, "chroma_path", str(tmp_path))
    monkeypatch.setattr(vt.settings, "chroma_host", "")
    monkeypatch.setattr(vt, "_client", None)

    def backend(name, dim):
        def f(t):
            v = [0.0] * dim
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
            n = sum(x * x for x in v) ** .5 or 1
            return [x / n for x in v]
        return E.EmbeddingBackend(name, "m", lambda ts: [f(t) for t in ts], f, dim)

    wm = {"id": "wm1", "titulo": "Arbolado", "snippet": "", "descripcion": "", "propietario": "x",
          "tags": [], "basemap": "", "extent": None, "total_capas": 1,
          "capas": [{"id": "l1", "titulo": "Encuesta arbolado", "tipo": "FL",
                     "url": "https://x/FeatureServer/0", "campos": []}]}
    monkeypatch.setattr(vt, "read_webmap", lambda i: dict(wm))

    monkeypatch.setattr(E, "_active", backend("viejo", 1024))
    vt.vectorize_webmap("wm1", deep=False)
    assert vt.search_webmap("arbolado", webmap_id="wm1")

    monkeypatch.setattr(E, "_active", backend("nuevo", 384))      # otra dimensión
    with pytest.raises(ValueError, match="Re-vectorizá"):
        vt.search_webmap("arbolado", webmap_id="wm1")

    r = vt.vectorize_webmap("wm1", deep=False)                     # se regenera solo
    assert r["embedding"] == "nuevo/m"
    assert vt.search_webmap("arbolado", webmap_id="wm1")[0]["metadata"]["webmap_id"] == "wm1"
    assert vt.search_webmap("arbolado", webmap_id="__layers__")


def test_numpy_embeddings_are_accepted_by_chroma(tmp_path, monkeypatch):
    """MiniLM local devuelve np.float32: debe poder guardarse en Chroma."""
    import numpy as np
    import llm.embeddings as E
    import tools.vector_tools as vt

    monkeypatch.setattr(vt.settings, "chroma_path", str(tmp_path))
    monkeypatch.setattr(vt.settings, "chroma_host", "")
    monkeypatch.setattr(vt, "_client", None)
    f = lambda t: np.array([len(t) % 7 + 1, 1, 2, 3], dtype=np.float32)
    monkeypatch.setattr(E, "_active", E.EmbeddingBackend(
        "local", "minilm", lambda ts: [list(f(t)) for t in ts], lambda t: list(f(t)), 4))
    monkeypatch.setattr(vt, "read_webmap", lambda i: {
        "id": "wm2", "titulo": "Cables", "snippet": "", "descripcion": "", "propietario": "x",
        "tags": [], "basemap": "", "extent": None, "total_capas": 1,
        "capas": [{"id": "c", "titulo": "Cables", "tipo": "FL", "url": "https://x/FeatureServer/0", "campos": []}]})
    assert vt.vectorize_webmap("wm2", deep=False)["documentos"] == 2
    assert vt.search_webmap("cables", webmap_id="wm2")


# ── Mapa abierto: el Planner decide entre describir y consultar ──

@pytest.fixture
def mapa_cables(tmp_path, monkeypatch, layer):
    """Vectoriza un WebMap de cables en un Chroma temporal con embeddings deterministas."""
    import hashlib
    import llm.embeddings as E
    import tools.vector_tools as vt

    monkeypatch.setattr(vt.settings, "chroma_path", str(tmp_path))
    monkeypatch.setattr(vt.settings, "chroma_host", "")
    monkeypatch.setattr(vt, "_client", None)

    def f(t):
        v = [0.0] * 64
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = sum(x * x for x in v) ** .5 or 1
        return [x / n for x in v]
    monkeypatch.setattr(E, "_active", E.EmbeddingBackend("test", "bow", lambda ts: [f(t) for t in ts], f, 64))

    monkeypatch.setattr(vt, "read_webmap", lambda i: {
        "id": "wmcables", "titulo": "Mapa Capacidad Cables", "snippet": "", "descripcion": "",
        "propietario": "x", "tags": [], "basemap": "", "extent": None, "total_capas": 3,
        "capas": [
            {"id": "g", "titulo": "Global Submarine Cables", "tipo": "GroupLayer", "url": "", "campos": []},
            {"id": "c", "titulo": "Cables", "tipo": "ArcGISFeatureLayer", "url": FakeLayer.url,
             "geometria": "esriGeometryPolyline", "total_features": 285,
             "campos": [{"nombre": "Name", "alias": "Name"}, {"nombre": "Capacity_G", "alias": "Capacidad (Gbps)"}]},
            {"id": "l", "titulo": "Landing Station", "tipo": "ArcGISFeatureLayer",
             "url": "https://x/landing/FeatureServer/0", "geometria": "esriGeometryPoint",
             "total_features": 737, "campos": [{"nombre": "Country", "alias": "Country"}]},
        ]})
    vt.vectorize_webmap("wmcables", deep=False)
    return "wmcables"


def test_with_open_map_questions_go_to_planner():
    from agents.dispatch import route
    assert route("¿qué capas tiene?", "wmcables")[0] == "Planner"
    assert route("rag: ¿qué capas tiene?", "wmcables")[0] == "RAGAgent"


def test_layers_brief_reads_index_without_llm(mapa_cables):
    from tools.vector_tools import webmap_layers_brief
    brief = webmap_layers_brief(mapa_cables)
    assert "Mapa Capacidad Cables" in brief and "3 capas" in brief
    assert FakeLayer.url in brief and "285 registros" in brief and "Capacity_G" in brief
    assert "sin URL" in brief                       # la capa de grupo queda marcada


def test_open_map_descriptive_question_uses_rag(mapa_cables, model):
    from agents.dispatch import dispatch
    m = model([
        text(json.dumps({"tareas": [{"agente": "RAGAgent", "tarea": "describir las capas del mapa"}]})),
        text("El mapa tiene 3 capas: Global Submarine Cables, Cables y Landing Station."),
    ])
    r = dispatch("¿Qué capas tiene este mapa?", webmap_id=mapa_cables, persist=False)
    assert r["agent_used"] == "Planner" and r["plan"][0]["agente"] == "RAGAgent"
    assert "3 capas" in r["answer"]
    assert {s["capa"] for s in r["sources"]} >= {"Cables"}         # capas citadas llegan a la UI
    assert "Landing Station" in str(m.seen[0])                     # el Planner vio las capas
    assert r["metrics"]["llm_calls"] == 2                          # planificar + redactar


def test_open_map_data_question_queries_the_layer(mapa_cables, model):
    from agents.dispatch import dispatch
    m = model([
        text(json.dumps({"tareas": [{"agente": "AnalyticsAgent",
             "tarea": f"Altura máxima en la capa Cables ({FakeLayer.url})"}]})),
        tool_call("field_statistics", {"layer_url": FakeLayer.url, "field_name": "_3_altura_total"}),
        text("El máximo es 32.5."),
    ])
    r = dispatch("¿cuál es el valor máximo?", webmap_id=mapa_cables, persist=False)
    assert r["plan"][0]["agente"] == "AnalyticsAgent"
    assert [s["tool"].split("·")[-1].strip() for s in r["steps"]] == ["field_statistics"]
    assert "32.5" in r["answer"]
    assert "MAPA ABIERTO" in str(m.seen[0]) and FakeLayer.url in str(m.seen[0])
