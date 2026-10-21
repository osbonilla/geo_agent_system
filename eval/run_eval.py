"""
eval/run_eval.py
────────────────
Evaluación SLM vs LLM sobre el mismo sistema multi-agente.

Corre cada pregunta de un archivo JSON con cada modelo y mide:

  ok_respuesta   la respuesta contiene el valor correcto
  ok_herramienta el agente llamó a las herramientas esperadas
  ok_agente      la pregunta llegó al agente esperado (vía Planner o prefijo)
  error          el sistema falló (excepción, formato, timeout)
  llm_calls, prompt_tokens, completion_tokens, duration_ms

La respuesta correcta puede darse de dos formas:
  · "respuesta_contiene": ["Eucalipto", "32.5"]           ← fija, a mano
  · "referencia": {"herramienta": "field_statistics",     ← se calcula EN VIVO
                   "args": {...}, "clave": "maximo"}         llamando a la herramienta
    Así la verdad siempre coincide con los datos actuales de AGOL.

Uso:
  python eval/run_eval.py --referencias        # revisa la respuesta correcta de cada pregunta
  python eval/run_eval.py --preguntas eval/preguntas.json \\
      --modelos ollama/qwen2.5:3b,ollama/qwen2.5:7b,nvidia_nim/openai/gpt-oss-20b \\
      --repeticiones 2

Resultados en eval/resultados/<fecha>/: detalle.csv, resumen.csv, resumen.md y
trazas.jsonl (plan, herramientas con argumentos y salidas: para analizar los errores)
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))

import logging  # noqa: E402
logging.basicConfig(level=logging.WARNING)


# ══════════════════════════════════════════════════════════
# COMPARACIÓN DE RESPUESTAS
# ══════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s)


_NUM = re.compile(r"-?\d[\d.,]*")


def _numbers(text: str):
    out = []
    for raw in _NUM.findall(text or ""):
        raw = raw.rstrip(".,")
        cands = {raw.replace(",", ""), raw.replace(".", "").replace(",", "."), raw.replace(",", ".")}
        for c in cands:
            try:
                out.append(float(c))
            except ValueError:
                pass
    return out


def contains_value(answer: str, expected, tol: float = 0.01) -> bool:
    """Texto: aparece (sin tildes ni mayúsculas). Número: aparece con tolerancia relativa."""
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        for n in _numbers(answer):
            if abs(n - expected) <= max(tol * abs(expected), 0.01) or round(n, 1) == round(expected, 1):
                return True
        return False
    return _norm(expected) in _norm(answer)


def reference_values(ref: dict) -> list:
    """Ejecuta la herramienta de referencia y extrae el valor esperado."""
    from tools import analytics_tools, arcgis_tools, postgis_tools, spatial_tools
    tools = {t.name: t for t in (analytics_tools.ANALYTICS_TOOLS + arcgis_tools.GEO_TOOLS
                                 + spatial_tools.SPATIAL_TOOLS + postgis_tools.QUERY_TOOLS)}
    args = dict(ref.get("args", {}))
    # "_geocodificar": "Quito" → lon/lat con el MISMO geocodificador que usa el agente.
    # "Quito" no es un punto: con coordenadas fijas, el más cercano depende de cuál se elija.
    lugar = args.pop("_geocodificar", None)
    if lugar:
        g = json.loads(tools["geocode_place"].invoke({"place": lugar}))
        if g.get("status") == "error":
            raise RuntimeError(f"no se pudo geocodificar '{lugar}': {g.get('mensaje')}")
        args.update(lon=g["lon"], lat=g["lat"])
    out = json.loads(tools[ref["herramienta"]].invoke(args))
    if out.get("status") == "error":
        raise RuntimeError(f"la referencia falló: {out.get('mensaje')}")
    val = out
    for key in str(ref["clave"]).split("."):
        val = val[int(key)] if isinstance(val, list) else val[key]
    return val if isinstance(val, list) else [val]


def tools_used(steps) -> set:
    return {str(s.get("tool", "")).split("·")[-1].strip() for s in steps}


def agents_used(r: dict) -> set:
    if r["agent_used"] != "Planner":
        return {r["agent_used"]}
    return {p["agente"] for p in r.get("plan", [])}


# ══════════════════════════════════════════════════════════
# EJECUCIÓN
# ══════════════════════════════════════════════════════════

def run(preguntas: list, modelos: list, repeticiones: int, outdir: str) -> list:
    from llm.router import set_llm
    from agents.dispatch import dispatch

    # La verdad se calcula una sola vez, antes de evaluar
    for q in preguntas:
        q["_esperado"] = list(q.get("respuesta_contiene", []))
        if q.get("referencia"):
            try:
                q["_esperado"] += reference_values(q["referencia"])
            except Exception as e:
                print(f"  ! referencia de {q['id']} no disponible: {e}")
                q["_ref_error"] = str(e)

    filas = []
    for modelo in modelos:
        print(f"\n=== {modelo} ===")
        try:
            set_llm(modelo)
        except Exception as e:
            print(f"  ! no se pudo activar: {e}")
            continue

        for rep in range(repeticiones):
            for q in preguntas:
                t0 = time.time()
                try:
                    r = dispatch(q["pregunta"], webmap_id=q.get("webmap_id"), persist=False)
                except Exception as e:
                    r = {"success": False, "answer": f"EXCEPCIÓN: {e}", "agent_used": "?",
                         "steps": [], "plan": [], "metrics": {"duration_ms": int((time.time() - t0) * 1000)}}

                esperado = q["_esperado"]
                ok_resp = bool(esperado) and all(contains_value(r["answer"], v) for v in esperado)
                herr = set(q.get("herramientas_esperadas", []))
                ok_herr = herr.issubset(tools_used(r["steps"])) if herr else None
                ag = q.get("agente_esperado")
                ok_ag = (ag in agents_used(r)) if ag else None
                m = r.get("metrics", {})

                fila = {
                    "modelo": modelo, "rep": rep + 1, "id": q["id"],
                    "categoria": q.get("categoria", ""), "pregunta": q["pregunta"],
                    "ok_respuesta": ok_resp if esperado else None,
                    "ok_herramienta": ok_herr, "ok_agente": ok_ag,
                    "error": not r.get("success", False),
                    "agente": r["agent_used"], "plan": json.dumps(r.get("plan", []), ensure_ascii=False),
                    "herramientas": ",".join(sorted(tools_used(r["steps"]))),
                    "llm_calls": m.get("llm_calls"), "prompt_tokens": m.get("prompt_tokens"),
                    "completion_tokens": m.get("completion_tokens"),
                    "duration_ms": m.get("duration_ms"), "grounding": m.get("grounding"),
                    "esperado": json.dumps(esperado, ensure_ascii=False),
                    "respuesta": r["answer"].replace("\n", " ")[:1000],
                }
                filas.append(fila)
                # Traza completa (plan + cada llamada a herramienta con sus argumentos y su
                # salida) para analizar POR QUÉ falló una pregunta, no solo que falló
                os.makedirs(outdir, exist_ok=True)
                with open(os.path.join(outdir, "trazas.jsonl"), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "modelo": modelo, "rep": rep + 1, "id": q["id"], "pregunta": q["pregunta"],
                        "ok_respuesta": fila["ok_respuesta"], "esperado": esperado,
                        "respuesta": r["answer"], "plan": r.get("plan", []),
                        "pasos": [{"tool": s.get("tool"), "input": s.get("input"),
                                   "output": str(s.get("output", ""))[:1500]} for s in r.get("steps", [])],
                    }, ensure_ascii=False, default=str) + "\n")
                marca = {True: "✓", False: "✗", None: "·"}[fila["ok_respuesta"]]
                print(f"  {marca} [{q['id']}] {fila['duration_ms']} ms · {fila['llm_calls']} llamadas"
                      f" · {fila['herramientas'] or '-'}")
    return filas


def _pct(vals):
    vals = [v for v in vals if v is not None]
    return round(100 * sum(1 for v in vals if v) / len(vals), 1) if vals else None


def _mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(statistics.mean(vals), 1) if vals else None


def summarize(filas: list) -> list:
    out = []
    for modelo in dict.fromkeys(f["modelo"] for f in filas):
        fs = [f for f in filas if f["modelo"] == modelo]
        dur = [f["duration_ms"] for f in fs if f["duration_ms"]]
        out.append({
            "modelo": modelo, "n": len(fs),
            "acierto_respuesta_%": _pct(f["ok_respuesta"] for f in fs),
            "acierto_herramienta_%": _pct(f["ok_herramienta"] for f in fs),
            "acierto_agente_%": _pct(f["ok_agente"] for f in fs),
            "tasa_error_%": _pct(f["error"] for f in fs),
            "latencia_media_s": round(_mean(dur) / 1000, 2) if dur else None,
            "latencia_p50_s": round(statistics.median(dur) / 1000, 2) if dur else None,
            "llamadas_llm_media": _mean(f["llm_calls"] for f in fs),
            "tokens_entrada_media": _mean(f["prompt_tokens"] for f in fs),
            "tokens_salida_media": _mean(f["completion_tokens"] for f in fs),
        })
    return out


def write(filas, resumen, outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, rows in (("detalle.csv", filas), ("resumen.csv", resumen)):
        if rows:
            with open(os.path.join(outdir, name), "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    if resumen:
        cols = list(resumen[0].keys())
        md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        md += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in resumen]
        cats = sorted({f["categoria"] for f in filas})
        md += ["", "### Acierto de respuesta por categoría (%)", "",
               "| modelo | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
        for r in resumen:
            vals = [_pct(f["ok_respuesta"] for f in filas
                         if f["modelo"] == r["modelo"] and f["categoria"] == c) for c in cats]
            md.append(f"| {r['modelo']} | " + " | ".join("" if v is None else str(v) for v in vals) + " |")
        with open(os.path.join(outdir, "resumen.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(md) + "\n")
        print("\n" + "\n".join(md))


def main():
    ap = argparse.ArgumentParser(description="Evaluación SLM vs LLM del sistema geoespacial")
    # Prioridad: preguntas.json (local, no versionado) → set de la tesis → ejemplo
    candidatos = ["preguntas.json", "preguntas_tesis.json", "preguntas.example.json"]
    default_q = next(p for p in (os.path.join(ROOT, "eval", c) for c in candidatos)
                     if os.path.exists(p) or p.endswith("example.json"))
    ap.add_argument("--preguntas", default=default_q)
    ap.add_argument("--modelos", default="", help="lista separada por comas, formato proveedor/modelo")
    ap.add_argument("--referencias", action="store_true",
                    help="solo calcula e imprime la respuesta correcta de cada pregunta (sin LLM)")
    ap.add_argument("--repeticiones", type=int, default=1)
    ap.add_argument("--solo", default="", help="ids de preguntas separados por coma (opcional)")
    ap.add_argument("--salida", default=os.path.join(ROOT, "eval", "resultados",
                                                     datetime.now().strftime("%Y%m%d_%H%M")))
    a = ap.parse_args()

    with open(a.preguntas, encoding="utf-8") as fh:
        preguntas = json.load(fh)["preguntas"]
    omitidas = [q["id"] for q in preguntas if "REEMPLAZAR" in json.dumps(q)]
    if omitidas:
        print(f"Se omiten (falta completar datos): {', '.join(omitidas)}")
    preguntas = [q for q in preguntas if q["id"] not in omitidas]
    if a.solo:
        ids = set(a.solo.split(","))
        preguntas = [q for q in preguntas if q["id"] in ids]

    if a.referencias:
        for q in preguntas:
            try:
                v = q.get("respuesta_contiene") or reference_values(q["referencia"])
                print(f"  ✓ {q['id']:<9} {q['pregunta'][:70]:<70} → {v}")
            except Exception as e:
                print(f"  ✗ {q['id']:<9} {q['pregunta'][:70]:<70} → ERROR: {e}")
        return
    modelos = [m.strip() for m in a.modelos.split(",") if m.strip()]
    if not modelos:
        ap.error("falta --modelos (o usa --referencias)")
    filas = run(preguntas, modelos, a.repeticiones, a.salida)
    write(filas, summarize(filas), a.salida)
    print(f"\nResultados en {a.salida}")


if __name__ == "__main__":
    main()
