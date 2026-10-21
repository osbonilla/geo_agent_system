"""
llm/router.py
──────────────
LLM Router — corazón del concepto SLM de la tesis.

Un único punto de entrada para todos los agentes:

    from llm.router import get_llm
    llm = get_llm()

Cómo elige el modelo
────────────────────
1. Si LLM_MODEL trae un proveedor explícito, se usa ese y solo ese.
   Si le falta la API key, falla temprano con un mensaje que dice qué hacer:

     LLM_MODEL='nvidia_nim/openai/gpt-oss-20b' usa el proveedor 'nvidia_nim',
     pero NVIDIA_NIM_API_KEY está vacío. Conseguí una key en
     https://build.nvidia.com y agregala a tu archivo .env.

2. Si LLM_MODEL=auto, recorre la cadena de prioridad:
     ollama (local) → nvidia_nim → groq → vpn → openai
   y se queda con el primero que responda y tenga credenciales.

Proveedores soportados: nvidia_nim · ollama · groq · openai · vpn
"""
import logging
import httpx
from typing import Optional
from langchain_core.language_models import BaseLanguageModel

from config.settings import get_settings
from llm.metrics import COUNTER
from llm.providers import (
    REGISTRY,
    AUTO_ORDER,
    parse_model_string,
    missing_key_error,
    known_providers,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Compatibilidad con el código anterior ─────────────────
class LLMProvider(str):
    OLLAMA     = "ollama"
    NVIDIA_NIM = "nvidia_nim"
    GROQ       = "groq"
    OPENAI     = "openai"
    VPN        = "vpn"


# ── Estado del router ─────────────────────────────────────
_active_provider : Optional[str] = None
_active_model    : Optional[str] = None
_active_llm      : Optional[BaseLanguageModel] = None
_last_error      : Optional[str] = None


# ══════════════════════════════════════════════════════════
# CREDENCIALES Y DISPONIBILIDAD
# ══════════════════════════════════════════════════════════

def _api_key(provider: str) -> str:
    return {
        "nvidia_nim": settings.nvidia_nim_api_key,
        "groq"      : settings.groq_api_key,
        "openai"    : settings.openai_api_key,
        "vpn"       : settings.vpn_api_key,
        "ollama"    : "local",
    }.get(provider, "")


def _default_model(provider: str) -> str:
    return {
        "nvidia_nim": REGISTRY["nvidia_nim"].default_model,
        "ollama"    : settings.ollama_model,
        "groq"      : settings.groq_model,
        "openai"    : settings.openai_model,
        "vpn"       : settings.vpn_model,
    }.get(provider, REGISTRY[provider].default_model)


def _has_credentials(provider: str) -> bool:
    if provider == "ollama":
        return True
    if provider == "vpn":
        return bool(settings.vpn_api_key and settings.vpn_api_base_url)
    return bool(_api_key(provider))


def _is_reachable(provider: str) -> bool:
    """Ping rápido — solo se usa en modo auto para no colgar el arranque."""
    try:
        if provider == "ollama":
            r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
            return r.status_code == 200
        if provider == "nvidia_nim":
            r = httpx.get(
                f"{settings.nvidia_nim_base_url}/models",
                headers={"Authorization": f"Bearer {settings.nvidia_nim_api_key}"},
                timeout=5.0,
            )
            return r.status_code == 200
        if provider == "vpn":
            r = httpx.get(
                f"{settings.vpn_api_base_url}/models",
                headers={"Authorization": f"Bearer {settings.vpn_api_key}"},
                timeout=3.0,
            )
            return r.status_code == 200
        # groq / openai: la key basta, no gastamos una llamada en el arranque
        return True
    except Exception as e:
        logger.debug(f"{provider} no responde: {e}")
        return False


# ══════════════════════════════════════════════════════════
# CONSTRUCTORES
# ══════════════════════════════════════════════════════════

def _build_nvidia_nim(model: str) -> BaseLanguageModel:
    """
    NVIDIA NIM (build.nvidia.com) — endpoint OpenAI-compatible.

    Usa ChatNVIDIA si langchain-nvidia-ai-endpoints está instalado,
    porque maneja mejor tool-calling y los modelos de razonamiento.
    Si no está, cae a ChatOpenAI apuntando al mismo endpoint.
    """
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(
            model=model,
            api_key=settings.nvidia_nim_api_key,
            base_url=settings.nvidia_nim_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            callbacks=[COUNTER],
        )
    except ImportError:
        logger.info("langchain-nvidia-ai-endpoints no instalado → usando ChatOpenAI")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=settings.nvidia_nim_api_key,
            base_url=settings.nvidia_nim_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
            callbacks=[COUNTER],
        )


def _build_ollama(model: str) -> BaseLanguageModel:
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.llm_max_tokens,
        callbacks=[COUNTER],
    )


def _build_groq(model: str) -> BaseLanguageModel:
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        callbacks=[COUNTER],
    )


def _build_openai(model: str) -> BaseLanguageModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        callbacks=[COUNTER],
    )


def _build_vpn(model: str) -> BaseLanguageModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        api_key=settings.vpn_api_key,
        base_url=settings.vpn_api_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        callbacks=[COUNTER],
    )


_BUILDERS = {
    "nvidia_nim": _build_nvidia_nim,
    "ollama"    : _build_ollama,
    "groq"      : _build_groq,
    "openai"    : _build_openai,
    "vpn"       : _build_vpn,
}


# ══════════════════════════════════════════════════════════
# API PÚBLICA
# ══════════════════════════════════════════════════════════

def get_llm(force_provider: str = None, force_model: str = None) -> BaseLanguageModel:
    """
    Retorna el LLM activo, construyéndolo la primera vez.

    Args:
        force_provider: ignora LLM_MODEL y usa este proveedor
        force_model   : modelo específico dentro del proveedor

    Raises:
        ValueError  : el proveedor pedido no tiene su API key configurada
        RuntimeError: ningún proveedor disponible en modo auto
    """
    global _active_provider, _active_model, _active_llm, _last_error

    if _active_llm is not None and not force_provider:
        return _active_llm

    provider, model = parse_model_string(settings.llm_model)

    if force_provider:
        provider = force_provider
        model = force_model or _default_model(force_provider)

    # ── Camino 1: proveedor explícito ─────────────────────
    if provider:
        if provider not in REGISTRY:
            raise ValueError(
                f"LLM_MODEL='{settings.llm_model}' usa el proveedor '{provider}', "
                f"que no existe. Proveedores válidos: {', '.join(REGISTRY)}."
            )
        if REGISTRY[provider].needs_key and not _api_key(provider):
            raise missing_key_error(settings.llm_model, provider)
        if provider == "vpn" and not settings.vpn_api_base_url:
            raise ValueError(
                f"LLM_MODEL='{settings.llm_model}' usa el proveedor 'vpn', pero "
                f"VPN_API_BASE_URL está vacío. Agregá la URL de tu endpoint "
                f"OpenAI-compatible a tu archivo .env."
            )

        model = model or _default_model(provider)
        llm = _BUILDERS[provider](model)
        _active_provider, _active_model, _active_llm, _last_error = provider, model, llm, None
        logger.info(f"LLM activo: {provider} — {model}")
        return llm

    # ── Camino 2: detección automática ────────────────────
    if not provider and model:
        # LLM_MODEL trae solo el nombre del modelo, sin proveedor
        logger.warning(
            f"LLM_MODEL='{settings.llm_model}' no indica proveedor. "
            f"Usá el formato <proveedor>/<modelo>, por ejemplo "
            f"nvidia_nim/openai/gpt-oss-20b. Detectando automáticamente..."
        )

    errors = []
    for candidate in AUTO_ORDER:
        if not _has_credentials(candidate):
            errors.append(f"{candidate}: sin credenciales")
            continue
        if not _is_reachable(candidate):
            errors.append(f"{candidate}: no responde")
            continue
        try:
            chosen_model = _default_model(candidate)
            llm = _BUILDERS[candidate](chosen_model)
            _active_provider, _active_model, _active_llm = candidate, chosen_model, llm
            _last_error = None
            logger.info(f"LLM activo (auto): {candidate} — {chosen_model}")
            return llm
        except Exception as e:
            errors.append(f"{candidate}: {e}")
            logger.warning(f"No se pudo construir {candidate}: {e}")

    _last_error = " · ".join(errors)
    raise RuntimeError(
        "No hay ningún LLM disponible. Configurá LLM_MODEL en tu .env, "
        "por ejemplo LLM_MODEL=nvidia_nim/openai/gpt-oss-20b con su "
        "NVIDIA_NIM_API_KEY (https://build.nvidia.com), o levantá Ollama "
        f"en {settings.ollama_base_url}. Detalle: {_last_error}"
    )


def get_llm_info() -> dict:
    """Información del LLM activo — la muestra la barra de estado del frontend."""
    try:
        get_llm()
        p = REGISTRY.get(_active_provider)
        return {
            "ok"       : True,
            "provider" : _active_provider,
            "label"    : p.label if p else _active_provider,
            "model"    : _active_model,
            "full"     : f"{_active_provider}/{_active_model}",
            "local"    : bool(p and p.local),
            "source"   : "LLM_MODEL" if parse_model_string(settings.llm_model)[0] else "auto",
        }
    except Exception as e:
        return {
            "ok"       : False,
            "provider" : None,
            "model"    : None,
            "error"    : str(e),
            "llm_model": settings.llm_model,
            "providers": known_providers(),
        }


def set_llm(model_string: str) -> dict:
    """
    Cambia el modelo en caliente desde la UI. Ejemplo:
        set_llm("nvidia_nim/openai/gpt-oss-20b")
    No escribe el .env: aplica solo mientras el proceso siga vivo.
    """
    global _active_llm
    provider, model = parse_model_string(model_string)
    if not provider:
        raise ValueError(
            f"'{model_string}' no tiene proveedor. Usá <proveedor>/<modelo>, "
            f"por ejemplo nvidia_nim/openai/gpt-oss-20b."
        )
    if REGISTRY[provider].needs_key and not _api_key(provider):
        raise missing_key_error(model_string, provider)

    _active_llm = None
    get_llm(force_provider=provider, force_model=model)
    return get_llm_info()


def reset_router() -> dict:
    """Fuerza la re-detección. Útil si Ollama arrancó después que la API."""
    global _active_provider, _active_model, _active_llm, _last_error
    _active_provider = _active_model = _active_llm = _last_error = None
    logger.info("Router reseteado — re-detectando LLM...")
    return get_llm_info()


def available_providers() -> list:
    """Proveedores del registro + si tienen credenciales listas."""
    out = []
    for p in known_providers():
        out.append({**p, "ready": _has_credentials(p["name"])})
    return out
