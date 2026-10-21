"""
llm/embeddings.py
─────────────────
Fábrica de embeddings para la vectorización de WebMaps.

Proveedores:
  ollama     → modelo local de embeddings (nomic-embed-text)
  local      → MiniLM que trae ChromaDB, sin API key (descarga ~80 MB la primera vez)
  nvidia_nim → NVIDIA NIM, requiere NVIDIA_NIM_API_KEY. Solo si se pide explícitamente.

EMBEDDING_PROVIDER=auto → ollama (si el modelo está descargado) → local.
NVIDIA no entra en "auto": un índice no debería depender de una API externa
que puede retirar el modelo (nv-embedqa-e5-v5 dejó de existir en agosto de
2026 y rompió todos los índices hechos con él).
La colección de Chroma guarda con qué proveedor fue vectorizada, así que
cambiar de proveedor implica re-vectorizar (los espacios vectoriales no
son intercambiables).
"""
import logging
from typing import List, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_active: Optional["EmbeddingBackend"] = None


class EmbeddingBackend:
    """Interfaz mínima: documentos y consultas."""

    def __init__(self, provider: str, model: str, fn_docs, fn_query, dim: Optional[int] = None):
        self.provider = provider
        self.model = model
        self._fn_docs = fn_docs
        self._fn_query = fn_query
        self.dim = dim

    # Se fuerza float de Python: MiniLM (onnx) devuelve np.float32 y Chroma
    # rechaza listas de escalares numpy ("Expected embeddings to be a list of floats").
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(x) for x in v] for v in self._fn_docs(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [float(x) for x in self._fn_query(text)]

    def info(self) -> dict:
        return {"provider": self.provider, "model": self.model, "dim": self.dim}


# ══════════════════════════════════════════════════════════
# BACKENDS
# ══════════════════════════════════════════════════════════

def _build_nvidia() -> EmbeddingBackend:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    emb = NVIDIAEmbeddings(
        model=settings.embedding_model,
        api_key=settings.nvidia_nim_api_key,
        base_url=settings.nvidia_nim_base_url,
        truncate="END",
    )
    return EmbeddingBackend(
        provider="nvidia_nim",
        model=settings.embedding_model,
        fn_docs=emb.embed_documents,
        fn_query=emb.embed_query,
    )


def _build_ollama() -> EmbeddingBackend:
    from langchain_ollama import OllamaEmbeddings
    emb = OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
    return EmbeddingBackend(
        provider="ollama",
        model=settings.ollama_embed_model,
        fn_docs=emb.embed_documents,
        fn_query=emb.embed_query,
    )


def _build_local() -> EmbeddingBackend:
    from chromadb.utils import embedding_functions
    fn = embedding_functions.DefaultEmbeddingFunction()
    return EmbeddingBackend(
        provider="local",
        model="all-MiniLM-L6-v2",
        fn_docs=lambda texts: [list(v) for v in fn(texts)],
        fn_query=lambda text: list(fn([text])[0]),
        dim=384,
    )


def _nvidia_ready() -> bool:
    if not settings.nvidia_nim_api_key:
        return False
    try:
        import langchain_nvidia_ai_endpoints  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "NVIDIA_NIM_API_KEY está configurada pero falta el paquete "
            "langchain-nvidia-ai-endpoints. Instalalo con: "
            "pip install langchain-nvidia-ai-endpoints"
        )
        return False


def _ollama_ready() -> bool:
    try:
        import httpx
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        if r.status_code != 200:
            return False
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return any(settings.ollama_embed_model in n for n in names)
    except Exception:
        return False


_BUILDERS = {
    "nvidia_nim": _build_nvidia,
    "ollama"    : _build_ollama,
    "local"     : _build_local,
}


def get_embeddings(force: str = None) -> EmbeddingBackend:
    """Retorna el backend de embeddings activo (se cachea tras la primera vez)."""
    global _active
    if _active is not None and not force:
        return _active

    choice = (force or settings.embedding_provider or "auto").lower()

    if choice == "auto":
        choice = "ollama" if _ollama_ready() else "local"

    if choice not in _BUILDERS:
        raise ValueError(
            f"EMBEDDING_PROVIDER='{choice}' no existe. "
            f"Opciones: nvidia_nim, ollama, local, auto."
        )

    if choice == "nvidia_nim" and not settings.nvidia_nim_api_key:
        raise ValueError(
            "EMBEDDING_PROVIDER='nvidia_nim' necesita NVIDIA_NIM_API_KEY. "
            "Conseguí una key en https://build.nvidia.com y agregala a tu .env."
        )

    try:
        backend = _BUILDERS[choice]()
    except Exception as e:
        if choice != "local":
            logger.warning(f"Embeddings '{choice}' fallaron ({e}) → usando 'local'")
            backend = _build_local()
        else:
            raise

    _active = backend
    logger.info(f"Embeddings: {backend.provider} — {backend.model}")
    return backend


def embedding_id() -> str:
    """Identificador del espacio vectorial: se guarda en cada colección de Chroma."""
    b = get_embeddings()
    return f"{b.provider}/{b.model}"


def get_embeddings_info() -> dict:
    try:
        return {"ok": True, **get_embeddings().info()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def reset_embeddings():
    global _active
    _active = None
