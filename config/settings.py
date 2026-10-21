"""
config/settings.py
──────────────────
Variables de entorno centralizadas con pydantic-settings.

Novedad v2:
  · LLM_MODEL usa la sintaxis  <proveedor>/<modelo>
      LLM_MODEL=nvidia_nim/openai/gpt-oss-20b
      LLM_MODEL=ollama/mistral
      LLM_MODEL=groq/llama-3.3-70b-versatile
      LLM_MODEL=auto          → detección automática por prioridad
  · Configuración de embeddings y de ChromaDB para la vectorización de WebMaps.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):

    # ── Selección de LLM ──────────────────────────────────
    # Formato: <proveedor>/<modelo>   ·   "auto" = detectar por prioridad
    llm_model       : str = Field("auto", env="LLM_MODEL")

    # ── NVIDIA NIM (build.nvidia.com) ─────────────────────
    nvidia_nim_api_key  : str = Field("", env="NVIDIA_NIM_API_KEY")
    nvidia_nim_base_url : str = Field(
        "https://integrate.api.nvidia.com/v1", env="NVIDIA_NIM_BASE_URL"
    )

    # ── Ollama (LLM local) ────────────────────────────────
    ollama_base_url : str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model    : str = Field("qwen2.5:7b", env="OLLAMA_MODEL")
    # Ventana de contexto de Ollama. Si no se fija, Ollama usa 2048-4096 tokens
    # y recorta el prompt EN SILENCIO: el SLM pierde instrucciones sin error.
    ollama_num_ctx  : int = Field(8192, env="OLLAMA_NUM_CTX")

    # ── API remota vía VPN (OpenAI-compatible) ────────────
    vpn_api_base_url: str = Field("", env="VPN_API_BASE_URL")
    vpn_api_key     : str = Field("", env="VPN_API_KEY")
    vpn_model       : str = Field("gpt-4o-mini", env="VPN_MODEL")

    # ── Groq ──────────────────────────────────────────────
    groq_api_key    : str = Field("", env="GROQ_API_KEY")
    groq_model      : str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")

    # ── OpenAI ────────────────────────────────────────────
    openai_api_key  : str = Field("", env="OPENAI_API_KEY")
    openai_model    : str = Field("gpt-4o-mini", env="OPENAI_MODEL")

    # ── Parámetros de inferencia ──────────────────────────
    llm_temperature : float = Field(0.0, env="LLM_TEMPERATURE")
    llm_max_tokens  : int   = Field(2048, env="LLM_MAX_TOKENS")
    llm_timeout     : int   = Field(120, env="LLM_TIMEOUT")

    # ── Embeddings (vectorización de WebMaps) ─────────────
    # auto (ollama → local) · ollama · local · nvidia_nim
    embedding_provider : str = Field("auto", env="EMBEDDING_PROVIDER")
    # Solo para EMBEDDING_PROVIDER=nvidia_nim (nv-embedqa-e5-v5 fue retirado)
    embedding_model    : str = Field("nvidia/llama-3.2-nv-embedqa-1b-v2", env="EMBEDDING_MODEL")
    ollama_embed_model : str = Field("nomic-embed-text", env="OLLAMA_EMBED_MODEL")

    # ── ChromaDB ──────────────────────────────────────────
    chroma_host       : str = Field("", env="CHROMA_HOST")     # vacío = modo local
    chroma_port       : int = Field(8001, env="CHROMA_PORT")
    chroma_path       : str = Field("./chroma_data", env="CHROMA_PATH")
    chroma_prefix     : str = Field("webmap", env="CHROMA_PREFIX")
    chroma_catalog    : str = Field("webmaps_catalog", env="CHROMA_CATALOG")
    # Índice global de capas (todas las capas de todos los WebMaps vectorizados).
    # Lo usa el grounding para darle al SLM URLs y campos reales.
    chroma_layers     : str = Field("webmap_layers", env="CHROMA_LAYERS")
    # WebMaps que no se listan ni se vectorizan (etiquetas separadas por coma).
    # Survey123 crea un WebMap con la etiqueta "Survey123" junto con cada formulario.
    webmaps_exclude_tags: str = Field("Survey123", env="WEBMAPS_EXCLUDE_TAGS")

    # ── Diseño para SLM ───────────────────────────────────
    grounding_k         : int   = Field(4, env="GROUNDING_K")
    grounding_min_sim   : float = Field(0.25, env="GROUNDING_MIN_SIM")
    agent_max_iterations: int   = Field(6, env="AGENT_MAX_ITERATIONS")

    # ── ArcGIS Online ─────────────────────────────────────
    arcgis_url      : str = Field("https://www.arcgis.com", env="ARCGIS_URL")
    arcgis_username : str = Field("", env="ARCGIS_USERNAME")
    arcgis_password : str = Field("", env="ARCGIS_PASSWORD")

    # ── PostgreSQL ────────────────────────────────────────
    db_host         : str = Field("localhost", env="DB_HOST")
    db_port         : int = Field(5432, env="DB_PORT")
    db_name         : str = Field("geo_agents_db", env="DB_NAME")
    db_user         : str = Field("postgres", env="DB_USER")
    db_password     : str = Field("postgres", env="DB_PASSWORD")

    # ── App ───────────────────────────────────────────────
    app_env         : str = Field("development", env="APP_ENV")
    log_level       : str = Field("INFO", env="LOG_LEVEL")
    api_port        : int = Field(8000, env="API_PORT")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def arcgis_configured(self) -> bool:
        return bool(self.arcgis_username and self.arcgis_password)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
