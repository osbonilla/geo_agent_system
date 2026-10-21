"""
llm/metrics.py
──────────────
Contador de llamadas y tokens al LLM, por petición.

Es la métrica de costo de la tesis: un diseño pensado para SLM debería
resolver la misma pregunta con menos llamadas y menos tokens.

    from llm.metrics import COUNTER
    COUNTER.reset()
    ... ejecutar agentes ...
    COUNTER.snapshot()  → {"llm_calls": 3, "prompt_tokens": 2100, "completion_tokens": 180}

Usa contextvars: cada petición HTTP (cada hilo del threadpool) tiene su cuenta.
"""
from contextvars import ContextVar
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler

_stats: ContextVar[Dict[str, int]] = ContextVar("llm_stats", default=None)


def _new() -> Dict[str, int]:
    return {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


class CallCounter(BaseCallbackHandler):
    def _get(self) -> Dict[str, int]:
        st = _stats.get()
        if st is None:
            st = _new()
            _stats.set(st)
        return st

    def reset(self) -> None:
        _stats.set(_new())

    def snapshot(self) -> Dict[str, int]:
        return dict(self._get())

    # Una llamada por cada invocación del modelo de chat
    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self._get()["llm_calls"] += 1

    def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
        self._get()["llm_calls"] += 1

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        st = self._get()
        try:
            for gens in response.generations:
                for g in gens:
                    um = getattr(getattr(g, "message", None), "usage_metadata", None)
                    if um:
                        st["prompt_tokens"] += int(um.get("input_tokens", 0) or 0)
                        st["completion_tokens"] += int(um.get("output_tokens", 0) or 0)
                        return
            usage = (response.llm_output or {}).get("token_usage") or {}
            st["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            st["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        except Exception:
            pass


COUNTER = CallCounter()
