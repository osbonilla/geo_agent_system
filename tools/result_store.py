"""
tools/result_store.py
─────────────────────
Canal lateral para las geometrías.

Problema: si una herramienta le devuelve al LLM 50 features con sus
geometrías completas, un polígono puede pesar miles de tokens. Un SLM con
8k de contexto se queda sin espacio y pierde las instrucciones.

Solución: la herramienta guarda acá las features completas (para el mapa
y para PostGIS) y al LLM le devuelve solo un resumen compacto. La API lee
este almacén al final de cada petición.

Usa contextvars: cada petición tiene su propio almacén.
"""
from contextvars import ContextVar
from typing import Dict, List, Optional

_store: ContextVar[Optional[List[Dict]]] = ContextVar("geo_results", default=None)

MAX_STORED = 2000


def reset() -> None:
    _store.set([])


def push(layer_url: str, features: List[Dict], layer_name: str = "") -> None:
    """features: [{"attributes": {...}, "geometry": {...ArcGIS JSON en 4326...}}]"""
    st = _store.get()
    if st is None:
        st = []
        _store.set(st)
    for f in features:
        if len(st) >= MAX_STORED:
            break
        st.append({"layer_url": layer_url, "layer_name": layer_name,
                   "attributes": f.get("attributes", {}), "geometry": f.get("geometry")})


def items() -> List[Dict]:
    return list(_store.get() or [])
