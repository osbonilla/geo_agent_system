"""
tests/fakes.py
──────────────
Dobles de prueba para correr el sistema sin ArcGIS, sin LLM y sin red:

  ScriptedToolModel → modelo de chat que devuelve respuestas guionadas,
                      incluidas llamadas a herramientas (tool calling)
  FakeLayer         → Feature Layer en memoria con 20 árboles de Quito
"""
import random
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# ── Modelo guionado ───────────────────────────────────────

def tool_call(name: str, args: dict, i: int = 0) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{i}"}],
                     usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})


def text(content: str) -> AIMessage:
    return AIMessage(content=content,
                     usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})


class ScriptedToolModel(BaseChatModel):
    script: List[AIMessage]
    seen: List[List[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.seen.append(messages)
        if not self.script:
            raise RuntimeError("el guion se quedó sin respuestas")
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


# ── Capa de ArcGIS en memoria ─────────────────────────────

class _Props(dict):
    __getattr__ = dict.get


class _Feature:
    def __init__(self, attributes, geometry=None):
        self.attributes, self.geometry = attributes, geometry


class _FS:
    def __init__(self, features):
        self.features = features


ESPECIES = ["Eucalipto"] * 8 + ["Arupo"] * 5 + ["Cholán"] * 4 + ["Pumamaqui"] * 3


def make_trees():
    rnd = random.Random(7)
    trees = []
    for i, esp in enumerate(ESPECIES, 1):
        trees.append({"OBJECTID": i, "field_7": esp,
                      "_3_altura_total": round(rnd.uniform(3, 30), 1) if i != 5 else 32.5,
                      "x": -78.4678 + rnd.uniform(-0.01, 0.01), "y": -0.1807 + rnd.uniform(-0.01, 0.01)})
    return trees


class FakeLayer:
    url = "https://services.example.com/arbolado/FeatureServer/0"

    def __init__(self):
        self.trees = make_trees()
        self.properties = _Props(
            name="Encuesta arbolado", geometryType="esriGeometryPoint",
            fields=[_Props(name="OBJECTID", alias="OBJECTID", type="esriFieldTypeOID"),
                    _Props(name="field_7", alias="Especie", type="esriFieldTypeString"),
                    _Props(name="_3_altura_total", alias="Altura total (m)", type="esriFieldTypeDouble")])
        self.calls = []

    def _rows(self, where="1=1", geometry_filter=None, distance=None, **_):
        rows = self.trees
        if "=" in where and where.strip() != "1=1" and "'" in where:
            val = where.split("'")[1]
            rows = [t for t in rows if t["field_7"] == val]
        if geometry_filter is not None and distance is not None:
            import math
            g = geometry_filter["geometry"]
            rows = [t for t in rows if math.hypot((t["x"] - g["x"]) * 111320,
                                                  (t["y"] - g["y"]) * 110574) <= distance]
        return rows

    def query(self, where="1=1", return_count_only=False, out_statistics=None,
              group_by_fields_for_statistics=None, order_by_fields=None,
              result_record_count=None, **kw):
        self.calls.append({"where": where, **kw})
        rows = self._rows(where, **kw)
        if return_count_only:
            return len(rows)
        if group_by_fields_for_statistics:
            f = group_by_fields_for_statistics
            counts = {}
            for t in rows:
                counts[t[f]] = counts.get(t[f], 0) + 1
            ordered = sorted(counts.items(), key=lambda kv: -kv[1])
            return _FS([_Feature({f: k, "conteo": v}) for k, v in ordered])
        if out_statistics:
            vals = [t[out_statistics[0]["onStatisticField"]] for t in rows]
            fn = {"min": min, "max": max, "sum": sum, "count": len,
                  "avg": lambda v: sum(v) / len(v)}
            return _FS([_Feature({s["outStatisticFieldName"]: fn[s["statisticType"]](vals)
                                  for s in out_statistics})])
        if order_by_fields:
            field, order = order_by_fields.split()
            rows = sorted(rows, key=lambda t: t[field], reverse=order.upper() == "DESC")
        rows = rows[: result_record_count or len(rows)]
        return _FS([_Feature({k: v for k, v in t.items() if k not in ("x", "y")},
                             {"x": t["x"], "y": t["y"]}) for t in rows])


# ── Capas genéricas para análisis espacial ────────────────

def square(x0, y0, x1, y1):
    """Anillo exterior en sentido horario (convención de Esri)."""
    return {"rings": [[[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]]]}


class FakeGeoLayer:
    """Feature Layer en memoria: features = [(atributos, geometría Esri)].
    Entiende filtros where del tipo  campo = 'valor'."""

    def __init__(self, url, name, geometry_type, fields, features):
        self.url = url
        self.properties = _Props(name=name, geometryType=geometry_type,
                                 fields=[_Props(name="OBJECTID", alias="OBJECTID", type="esriFieldTypeOID")]
                                 + [_Props(name=n, alias=a, type=t) for n, a, t in fields])
        self.features = [({"OBJECTID": i, **a}, g) for i, (a, g) in enumerate(features, 1)]
        self.calls = 0

    def query(self, where="1=1", return_count_only=False, **kw):
        self.calls += 1
        self.last_kw = kw
        rows = self.features
        w = (where or "1=1").strip()
        if w != "1=1":
            campo, valor = [p.strip() for p in w.split("=", 1)]
            rows = [r for r in rows if str(r[0].get(campo)) == valor.strip("'")]
        if return_count_only:
            return len(rows)
        campo = kw.get("group_by_fields_for_statistics")
        if campo:
            conteo = {}
            for a, _ in rows:
                conteo[a.get(campo)] = conteo.get(a.get(campo), 0) + 1
            return _FS([_Feature({campo: k, "conteo": v}) for k, v in
                        sorted(conteo.items(), key=lambda kv: -kv[1])])
        return _FS([_Feature(dict(a), g) for a, g in rows])
