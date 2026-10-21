"""
eval/diagnostico_tools.py
─────────────────────────
¿El modelo hace tool calling nativo a través de este sistema?

Separa dos causas que en la evaluación se ven igual (0 herramientas llamadas):
  · falla de INTEGRACIÓN: el modelo nunca devuelve tool_calls, ni con una
    herramienta trivial y un prompt mínimo → problema de proveedor/plantilla
  · falla de COMPORTAMIENTO: con el caso trivial sí llama, pero con el prompt
    real del agente describe la llamada en texto → limitación del modelo

Uso:
  docker compose exec api python eval/diagnostico_tools.py --modelos ollama/qwen2.5:7b,ollama/mistral:7b
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402


@tool
def sumar(a: int, b: int) -> int:
    """Suma dos números enteros."""
    return a + b


CASOS = [
    ("trivial (inglés)", None, "What is 2 + 3? Use the sumar tool."),
    ("trivial (español)", None, "¿Cuánto es 2 + 3? Usa la herramienta sumar."),
    ("prompt real del SpatialAgent", "spatial",
     "Cuenta los puntos de https://x/reportes/FeatureServer/0 con filtro intencion_bert = 'denuncia' "
     "dentro de cada polígono de https://x/parroquias/FeatureServer/2."),
]


def probar(modelo: str) -> None:
    from llm.router import set_llm, get_llm
    set_llm(modelo)
    llm = get_llm()
    print(f"\n=== {modelo} ===")
    for nombre, sistema, pregunta in CASOS:
        if sistema == "spatial":
            from agents.spatial_agent import SPATIAL_SYSTEM, TOOLS
            mensajes = [SystemMessage(SPATIAL_SYSTEM), HumanMessage(pregunta)]
            herramientas = TOOLS
        else:
            mensajes = [HumanMessage(pregunta)]
            herramientas = [sumar]
        try:
            r = llm.bind_tools(herramientas).invoke(mensajes)
            llamadas = [c["name"] for c in (r.tool_calls or [])]
            texto = (r.content if isinstance(r.content, str) else str(r.content)).strip().replace("\n", " ")
            marca = "✓" if llamadas else "✗"
            print(f"  {marca} {nombre:<30} tool_calls={llamadas or '—'}")
            if not llamadas:
                print(f"      texto: {texto[:220]}")
        except Exception as e:
            print(f"  ! {nombre:<30} ERROR: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", required=True)
    for m in ap.parse_args().modelos.split(","):
        if m.strip():
            probar(m.strip())


if __name__ == "__main__":
    main()
