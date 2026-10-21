"""
llm/structured.py
─────────────────
Salida estructurada para SLM.

Un SLM que tiene que "escribir JSON válido" a mano se equivoca seguido:
comillas, markdown alrededor, claves inventadas. Acá se le pide al proveedor
que FUERCE el esquema (json_schema en Ollama/OpenAI, guided JSON en NIM),
de modo que la respuesta es válida por construcción.

Si el modelo o el proveedor no soporta salida estructurada, se cae a pedir
texto, extraer el primer objeto JSON y validarlo con Pydantic.

    plan = invoke_structured(PROMPT, {"user_input": "..."}, Plan)
"""
import json
import logging
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from llm.router import get_llm, get_llm_info

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"la respuesta no contiene JSON: {text[:200]}")
        return json.loads(m.group(0))


def invoke_structured(prompt, variables: dict, schema: Type[T]) -> T:
    """Invoca prompt | llm forzando el esquema; con respaldo a parseo de texto."""
    llm = get_llm()
    provider = get_llm_info().get("provider")

    raw = None
    try:
        kwargs = {"include_raw": True}
        if provider in ("ollama", "openai", "vpn"):
            kwargs["method"] = "json_schema"
        out = (prompt | llm.with_structured_output(schema, **kwargs)).invoke(variables)
        if isinstance(out, dict) and "raw" in out:
            if out.get("parsed") is not None:
                parsed = out["parsed"]
                return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
            raw = out.get("raw")   # el modelo respondió pero no con el esquema
        elif isinstance(out, schema):
            return out
        elif isinstance(out, dict):
            return schema.model_validate(out)
    except Exception as e:
        logger.info(f"Salida estructurada no disponible ({e.__class__.__name__}: {e}); "
                    f"usando parseo de texto")

    # Si ya hubo respuesta se reutiliza: con un SLM local cada llamada cuesta segundos
    if raw is None:
        raw = (prompt | llm).invoke(variables)
    content = raw.content if hasattr(raw, "content") else str(raw)
    if isinstance(content, list):
        content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    try:
        return schema.model_validate(_extract_json(content))
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"El modelo no devolvió un {schema.__name__} válido: {e}") from e
