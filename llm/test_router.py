"""
llm/test_router.py
───────────────────
Verifica qué modelo va a usar el sistema y que responda de verdad.

Uso:
  python llm/test_router.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def test_router():
    print("\n" + "=" * 60)
    print("TEST: LLM Router")
    print("=" * 60)

    from config.settings import get_settings
    from llm.router import get_llm, get_llm_info, available_providers

    s = get_settings()
    print(f"\nLLM_MODEL en el .env: {s.llm_model}")

    # ── 1. Proveedores del registro ───────────────────────
    print("\n1. Proveedores soportados")
    for p in available_providers():
        estado = "listo" if p["ready"] else f"falta {p['env_key']}"
        print(f"   {p['name']:<12} {estado:<28} LLM_MODEL={p['example']}")

    # ── 2. Resolución del modelo ──────────────────────────
    print("\n2. Resolviendo el modelo activo")
    try:
        llm = get_llm()
        info = get_llm_info()
        print(f"   Proveedor : {info['provider']} ({info['label']})")
        print(f"   Modelo    : {info['model']}")
        print(f"   Local     : {'sí' if info['local'] else 'no'}")
        print(f"   Elegido por: {info['source']}")
    except Exception as e:
        # Acá aparece, por ejemplo, el aviso de NVIDIA_NIM_API_KEY vacía
        print(f"\n   {e}\n")
        return False

    # ── 3. Invocación real ────────────────────────────────
    print("\n3. Invocando el modelo")
    try:
        response = llm.invoke("Responde solo con: OK")
        content = response.content if hasattr(response, "content") else str(response)
        print(f"   Respuesta: {content.strip()[:60]}")
    except Exception as e:
        print(f"   Error al invocar: {e}")
        return False

    # ── 4. Embeddings ─────────────────────────────────────
    print("\n4. Embeddings para vectorizar WebMaps")
    try:
        from llm.embeddings import get_embeddings
        emb = get_embeddings()
        vector = emb.embed_query("capa de arbolado urbano de Quito")
        print(f"   Backend   : {emb.provider} — {emb.model}")
        print(f"   Dimensión : {len(vector)}")
    except Exception as e:
        print(f"   Embeddings no disponibles: {e}")

    print("\nRouter funcionando correctamente.\n")
    return True


if __name__ == "__main__":
    sys.exit(0 if test_router() else 1)
