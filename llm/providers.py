"""
llm/providers.py
────────────────
Registro de proveedores de LLM.

Cada proveedor declara:
  · qué variable de entorno guarda su API key
  · dónde se consigue esa key
  · su base_url por defecto
  · cómo construir el objeto LangChain

Sintaxis de LLM_MODEL:  <proveedor>/<modelo>
  nvidia_nim/openai/gpt-oss-20b   → proveedor nvidia_nim, modelo openai/gpt-oss-20b
  ollama/mistral                  → proveedor ollama,     modelo mistral
  groq/llama-3.3-70b-versatile    → proveedor groq,       modelo llama-3.3-70b-versatile
  auto                            → detección automática
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class Provider:
    name        : str            # identificador en LLM_MODEL
    label       : str            # nombre legible para la UI
    env_key     : str            # variable de entorno con la API key ("" si no requiere)
    signup_url  : str            # dónde conseguir la key
    default_model: str
    local       : bool = False   # corre en la máquina del usuario
    base_url_env: str = ""

    @property
    def needs_key(self) -> bool:
        return bool(self.env_key)


NVIDIA_NIM = Provider(
    name          = "nvidia_nim",
    label         = "NVIDIA NIM",
    env_key       = "NVIDIA_NIM_API_KEY",
    signup_url    = "https://build.nvidia.com",
    default_model = "openai/gpt-oss-20b",
    base_url_env  = "NVIDIA_NIM_BASE_URL",
)

OLLAMA = Provider(
    name          = "ollama",
    label         = "Ollama local",
    env_key       = "",
    signup_url    = "https://ollama.com/download",
    default_model = "qwen2.5:7b",
    local         = True,
    base_url_env  = "OLLAMA_BASE_URL",
)

GROQ = Provider(
    name          = "groq",
    label         = "Groq",
    env_key       = "GROQ_API_KEY",
    signup_url    = "https://console.groq.com/keys",
    default_model = "llama-3.3-70b-versatile",
)

OPENAI = Provider(
    name          = "openai",
    label         = "OpenAI",
    env_key       = "OPENAI_API_KEY",
    signup_url    = "https://platform.openai.com/api-keys",
    default_model = "gpt-4o-mini",
)

VPN = Provider(
    name          = "vpn",
    label         = "API remota (VPN, OpenAI-compatible)",
    env_key       = "VPN_API_KEY",
    signup_url    = "el administrador de tu VPN",
    default_model = "gpt-4o-mini",
    base_url_env  = "VPN_API_BASE_URL",
)


REGISTRY = {p.name: p for p in (NVIDIA_NIM, OLLAMA, GROQ, OPENAI, VPN)}

# Orden de detección automática cuando LLM_MODEL=auto
AUTO_ORDER = ["ollama", "nvidia_nim", "groq", "vpn", "openai"]


def parse_model_string(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Separa 'nvidia_nim/openai/gpt-oss-20b' en ('nvidia_nim', 'openai/gpt-oss-20b').

    Retorna (None, None) si el valor es 'auto', vacío, o no trae proveedor válido.
    El modelo puede contener '/' — solo el primer segmento es el proveedor.
    """
    raw = (raw or "").strip()
    if not raw or raw.lower() in ("auto", "detect", "automatico", "automático"):
        return None, None

    if "/" not in raw:
        # LLM_MODEL=mistral  →  sin proveedor explícito
        return None, raw

    provider, model = raw.split("/", 1)
    provider = provider.strip().lower()
    model = model.strip()

    if provider not in REGISTRY:
        return None, raw

    return provider, (model or REGISTRY[provider].default_model)


def missing_key_error(llm_model: str, provider_name: str) -> ValueError:
    """
    Construye el error explicativo cuando falta la API key del proveedor elegido.

    Ejemplo de mensaje:
      LLM_MODEL='nvidia_nim/openai/gpt-oss-20b' usa el proveedor 'nvidia_nim',
      pero NVIDIA_NIM_API_KEY está vacío. Conseguí una key en
      https://build.nvidia.com y agregala a tu archivo .env.
    """
    p = REGISTRY[provider_name]
    return ValueError(
        f"LLM_MODEL='{llm_model}' usa el proveedor '{p.name}', pero "
        f"{p.env_key} está vacío. Conseguí una key en {p.signup_url} "
        f"y agregala a tu archivo .env."
    )


def known_providers() -> list:
    """Lista serializable de proveedores — la consume el frontend."""
    return [
        {
            "name"         : p.name,
            "label"        : p.label,
            "env_key"      : p.env_key,
            "signup_url"   : p.signup_url,
            "default_model": p.default_model,
            "local"        : p.local,
            "example"      : f"{p.name}/{p.default_model}",
        }
        for p in REGISTRY.values()
    ]
