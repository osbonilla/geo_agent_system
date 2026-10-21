"""
tools/vector_tools.py
─────────────────────
Vectorización de WebMaps de ArcGIS Online.

Qué hace
────────
Toma un WebMap de AGOL, lo desarma en piezas con sentido semántico
(ficha del mapa, cada capa operacional, cada campo relevante), las
convierte en embeddings y las guarda en ChromaDB.

Después el RAGAgent puede responder "¿qué capa tiene datos de arbolado?"
sin recorrer la API de Esri: busca por similitud en el índice.

Colecciones
───────────
  webmap_<item_id>   → una por WebMap, con sus capas y campos
  webmaps_catalog    → índice global, una entrada por WebMap

Modo de conexión a Chroma
─────────────────────────
  CHROMA_HOST vacío  → PersistentClient en CHROMA_PATH (archivo local)
  CHROMA_HOST puesto → HttpClient contra el contenedor chroma
"""
import re
import json
import logging
from typing import Optional, List, Dict, Any

from langchain_core.tools import tool

from config.settings import get_settings
from llm.embeddings import get_embeddings, embedding_id

logger = logging.getLogger(__name__)
settings = get_settings()

_client = None


# ══════════════════════════════════════════════════════════
# CLIENTE CHROMA
# ══════════════════════════════════════════════════════════

def get_chroma_client():
    """Cliente de ChromaDB — HTTP si hay CHROMA_HOST, si no en disco."""
    global _client
    if _client is not None:
        return _client

    import chromadb

    if settings.chroma_host:
        logger.info(f"ChromaDB remoto: {settings.chroma_host}:{settings.chroma_port}")
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    else:
        logger.info(f"ChromaDB local: {settings.chroma_path}")
        _client = chromadb.PersistentClient(path=settings.chroma_path)

    return _client


def collection_name(item_id: str) -> str:
    """
    Nombre de colección válido para Chroma:
    3-63 caracteres, alfanuméricos, guion y guion bajo, empieza y termina en alfanumérico.
    """
    clean = re.sub(r"[^a-zA-Z0-9_-]", "", str(item_id))[:40]
    return f"{settings.chroma_prefix}_{clean}"


def _get_collection(name: str, emb: Optional[str] = None):
    """
    Colección para ESCRIBIR. Guarda en su metadata con qué embeddings se creó.
    Si existía con otros embeddings (u otra dimensión), se borra y se recrea:
    mezclar espacios vectoriales da resultados sin sentido o errores de dimensión.
    """
    emb = emb or embedding_id()
    client = get_chroma_client()
    try:
        col = client.get_collection(name)
    except Exception:
        col = None
    if col is not None and (col.metadata or {}).get("embedding") != emb:
        logger.warning(f"Colección {name} hecha con {(col.metadata or {}).get('embedding')}; "
                       f"se regenera con {emb}")
        client.delete_collection(name)
        col = None
    if col is None:
        col = client.create_collection(name=name, metadata={"hnsw:space": "cosine", "embedding": emb})
    return col


# ══════════════════════════════════════════════════════════
# LECTURA DE WEBMAPS DESDE ARCGIS ONLINE
# ══════════════════════════════════════════════════════════

def list_webmaps(max_items: int = 50, query: str = None) -> List[dict]:
    """Lista los WebMaps visibles para el usuario conectado."""
    from tools.arcgis_tools import get_gis_connection

    gis = get_gis_connection()
    q = query or f"owner:{gis.properties.user.username}"
    items = gis.content.search(query=q, item_type="Web Map", max_items=max_items)

    excluir = {t.strip().lower() for t in settings.webmaps_exclude_tags.split(",") if t.strip()}
    out = []
    for it in items:
        if excluir & {str(t).lower() for t in (it.tags or [])}:
            continue   # p. ej. el WebMap que Survey123 crea junto con el formulario
        out.append({
            "id"        : it.id,
            "titulo"    : it.title,
            "snippet"   : it.snippet or "",
            "propietario": it.owner,
            "tags"      : it.tags or [],
            "modificado": getattr(it, "modified", None),
            "thumbnail" : it.thumbnail or "",
            "url_item"  : f"{settings.arcgis_url}/home/item.html?id={it.id}",
        })
    return out


def _walk_layers(layers: List[dict], parent: str = "") -> List[dict]:
    """Aplana capas operacionales, incluidas las agrupadas."""
    flat = []
    for lyr in layers or []:
        entry = {
            "id"        : lyr.get("id", ""),
            "titulo"    : lyr.get("title", "Sin título"),
            "tipo"      : lyr.get("layerType", ""),
            "url"       : lyr.get("url", ""),
            "item_id"   : lyr.get("itemId", ""),
            "visible"   : lyr.get("visibility", True),
            "opacidad"  : lyr.get("opacity", 1),
            "grupo"     : parent,
            "campos"    : [],
        }

        popup = lyr.get("popupInfo") or {}
        for f in popup.get("fieldInfos", []) or []:
            entry["campos"].append({
                "nombre": f.get("fieldName", ""),
                "alias" : f.get("label", ""),
            })

        flat.append(entry)

        if lyr.get("layers"):
            flat.extend(_walk_layers(lyr["layers"], parent=entry["titulo"]))

    return flat


def read_webmap(item_id: str) -> dict:
    """Lee la definición completa de un WebMap: ficha + capas + basemap."""
    from tools.arcgis_tools import get_gis_connection

    gis = get_gis_connection()
    item = gis.content.get(str(item_id).strip())
    if item is None:
        raise ValueError(f"El WebMap '{item_id}' no existe o no es accesible con este usuario.")

    data = item.get_data() or {}
    capas = _walk_layers(data.get("operationalLayers", []))
    basemap = (data.get("baseMap") or {}).get("title", "")

    return {
        "id"         : item.id,
        "titulo"     : item.title,
        "snippet"    : item.snippet or "",
        "descripcion": re.sub(r"<[^>]+>", " ", item.description or "").strip(),
        "propietario": item.owner,
        "tags"       : item.tags or [],
        "basemap"    : basemap,
        "extent"     : item.extent,
        "capas"      : capas,
        "total_capas": len(capas),
    }


def _enrich_fields(capa: dict) -> dict:
    """Completa campos y conteo consultando el Feature Layer (best effort)."""
    url = capa.get("url", "")
    if not url or "FeatureServer" not in url:
        return capa
    try:
        from arcgis.features import FeatureLayer
        from tools.arcgis_tools import get_gis_connection

        layer = FeatureLayer(url, gis=get_gis_connection())
        props = layer.properties
        capa["geometria"] = props.get("geometryType", "")
        capa["campos"] = [
            {"nombre": f.get("name"), "alias": f.get("alias"), "tipo": f.get("type")}
            for f in props.get("fields", [])
        ]
        capa["total_features"] = layer.query(return_count_only=True)
        capa["valores"] = category_values(layer)
    except Exception as e:
        logger.debug(f"No se pudieron leer campos de {url}: {e}")
    return capa


MAX_VALORES = 12          # un campo con más valores distintos no es categórico
MAX_CAMPOS_VALORES = 40   # campos de texto revisados por capa (la de accidentes tiene ~150)
_TEXTO_LIBRE = ("descrip", "coment", "observ", "direcc", "detalle", "nombredelavia", "siniestro",
                "codigo", "globalid", "url", "foto", "imagen")


def category_values(layer) -> Dict[str, List[str]]:
    """
    Valores posibles de los campos categóricos: {"intencion_bert": ["queja", "denuncia", …]}.

    Por qué: la evaluación mostró que, ante "¿qué parroquia tiene más DENUNCIAS?", el
    modelo contaba TODOS los reportes. Sin ver los valores no sabe que "denuncias" se
    traduce en intencion_bert = 'denuncia' ni que "seguridad" es un valor de categoria.
    Se toman de los dominios codificados del servicio o, si no hay, de un conteo
    agrupado (una consulta por campo, solo al vectorizar).
    """
    from tools.arcgis_tools import _norm, SYSTEM_FIELDS
    out: Dict[str, List[str]] = {}
    revisados = 0
    for f in layer.properties.get("fields", []):
        nombre = f.get("name", "")
        if f.get("type") != "esriFieldTypeString" or _norm(nombre) in SYSTEM_FIELDS:
            continue
        if any(t in _norm(nombre) or t in _norm(f.get("alias") or "") for t in _TEXTO_LIBRE):
            continue
        dominio = (f.get("domain") or {}).get("codedValues") or []
        if dominio:
            vals = [str(c.get("code")) for c in dominio]
        else:
            if revisados >= MAX_CAMPOS_VALORES:
                continue
            revisados += 1
            try:
                fs = layer.query(where="1=1", group_by_fields_for_statistics=nombre,
                                 out_statistics=[{"statisticType": "count", "onStatisticField": nombre,
                                                  "outStatisticFieldName": "conteo"}])
                vals = [str(a.get(nombre)) for a in (ft.attributes for ft in fs.features)
                        if a.get(nombre) not in (None, "")]
            except Exception:
                continue
        if 1 < len(vals) <= MAX_VALORES:
            out[nombre] = vals
    return out


def indexed_values(layer_url: str) -> str:
    """Línea de valores guardada al vectorizar para esta capa ("" si no está indexada).
    Se lee del índice global de capas: no cuesta consultas a ArcGIS ni embeddings."""
    try:
        col = get_chroma_client().get_collection(settings.chroma_layers)
        data = col.get(where={"capa_url": layer_url}, include=["documents"])
    except Exception:
        return ""
    for doc in data.get("documents") or []:
        m = re.search(r"^Valores: (.*)$", doc or "", re.MULTILINE)
        if m:
            return m.group(1)
    return ""


def format_values(valores: Dict[str, List[str]], max_chars: int = 700) -> str:
    """{"categoria": ["basura", "seguridad"]} → "categoria = basura|seguridad; …" """
    partes, total = [], 0
    for campo, vals in (valores or {}).items():
        t = f"{campo} = {'|'.join(vals)}"
        if total + len(t) > max_chars:
            break
        partes.append(t)
        total += len(t) + 2
    return "; ".join(partes)


# ══════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE DOCUMENTOS
# ══════════════════════════════════════════════════════════

def build_documents(webmap: dict) -> List[Dict[str, Any]]:
    """
    Convierte un WebMap en documentos indexables.
    Un documento = una unidad que tiene sentido recuperar sola.
    """
    docs = []
    wid = webmap["id"]

    ficha = (
        f"WebMap: {webmap['titulo']}\n"
        f"Resumen: {webmap['snippet']}\n"
        f"Descripción: {webmap['descripcion'][:1500]}\n"
        f"Propietario: {webmap['propietario']}\n"
        f"Etiquetas: {', '.join(webmap['tags'])}\n"
        f"Mapa base: {webmap['basemap']}\n"
        f"Capas ({webmap['total_capas']}): "
        + ", ".join(c["titulo"] for c in webmap["capas"])
    )
    docs.append({
        "id"      : f"{wid}::ficha",
        "text"    : ficha,
        "metadata": {
            "webmap_id"    : wid,
            "webmap_titulo": webmap["titulo"],
            "tipo"         : "ficha",
            "capa_titulo"  : "",
            "capa_url"     : "",
        },
    })

    for i, capa in enumerate(webmap["capas"]):
        campos = ", ".join(
            f"{c.get('nombre')} ({c.get('alias') or c.get('tipo') or ''})".strip()
            for c in capa.get("campos", [])[:60]
        )
        texto = (
            f"Capa: {capa['titulo']}\n"
            f"WebMap: {webmap['titulo']}\n"
            f"Tipo: {capa['tipo']}  Geometría: {capa.get('geometria', 'n/d')}\n"
            f"Grupo: {capa.get('grupo') or 'ninguno'}\n"
            f"Registros: {capa.get('total_features', 'n/d')}\n"
            f"URL: {capa['url']}\n"
            + (f"Valores: {format_values(capa.get('valores'))}\n" if capa.get("valores") else "")
            + f"Campos: {campos or 'no publicados'}"
        )
        docs.append({
            "id"      : f"{wid}::capa::{capa['id'] or i}",
            "text"    : texto,
            "metadata": {
                "webmap_id"     : wid,
                "webmap_titulo" : webmap["titulo"],
                "tipo"          : "capa",
                "capa_titulo"   : capa["titulo"],
                "capa_url"      : capa["url"] or "",
                "capa_item_id"  : capa.get("item_id", "") or "",
                "geometria"     : capa.get("geometria", "") or "",
                "total_features": int(capa.get("total_features") or 0),
            },
        })

    return docs


# ══════════════════════════════════════════════════════════
# VECTORIZACIÓN
# ══════════════════════════════════════════════════════════

def vectorize_webmap(item_id: str, deep: bool = True) -> dict:
    """
    Vectoriza un WebMap completo y lo deja consultable.

    deep=True consulta cada Feature Layer para traer campos y conteo real.
    Es más lento pero produce respuestas mucho mejores.
    """
    webmap = read_webmap(item_id)

    if deep:
        webmap["capas"] = [_enrich_fields(c) for c in webmap["capas"]]

    docs = build_documents(webmap)
    emb = get_embeddings()
    vectors = emb.embed_documents([d["text"] for d in docs])

    col = _get_collection(collection_name(item_id))
    # Se borra lo anterior para no dejar capas que ya no están en el mapa
    try:
        col.delete(where={"webmap_id": webmap["id"]})
    except Exception:
        pass
    col.upsert(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        embeddings=vectors,
        metadatas=[d["metadata"] for d in docs],
    )

    # Índice global de capas: lo usa el grounding de los agentes
    capas_docs = [(d, v) for d, v in zip(docs, vectors) if d["metadata"]["tipo"] == "capa"
                  and d["metadata"]["capa_url"]]
    layers_col = _get_collection(settings.chroma_layers)
    try:
        layers_col.delete(where={"webmap_id": webmap["id"]})
    except Exception:
        pass
    if capas_docs:
        layers_col.upsert(
            ids=[d["id"] for d, _ in capas_docs],
            documents=[d["text"] for d, _ in capas_docs],
            embeddings=[v for _, v in capas_docs],
            metadatas=[d["metadata"] for d, _ in capas_docs],
        )

    # Índice global: una entrada por WebMap
    catalog = _get_collection(settings.chroma_catalog)
    catalog.upsert(
        ids=[webmap["id"]],
        documents=[docs[0]["text"]],
        embeddings=[vectors[0]],
        metadatas=[{
            "webmap_id"    : webmap["id"],
            "webmap_titulo": webmap["titulo"],
            "propietario"  : webmap["propietario"],
            "total_capas"  : webmap["total_capas"],
            "coleccion"    : collection_name(item_id),
            "embedding"    : f"{emb.provider}/{emb.model}",
            "tags"         : ", ".join(webmap["tags"]),
        }],
    )

    logger.info(
        f"WebMap vectorizado: {webmap['titulo']} — "
        f"{len(docs)} documentos, {webmap['total_capas']} capas"
    )

    return {
        "ok"          : True,
        "webmap_id"   : webmap["id"],
        "titulo"      : webmap["titulo"],
        "coleccion"   : collection_name(item_id),
        "documentos"  : len(docs),
        "capas"       : webmap["total_capas"],
        "embedding"   : f"{emb.provider}/{emb.model}",
        "profundo"    : deep,
    }


def vectorize_all_webmaps(max_items: int = 20, deep: bool = False) -> dict:
    """Vectoriza todos los WebMaps del usuario. Pensado para correr una vez."""
    resultados, errores = [], []
    for wm in list_webmaps(max_items=max_items):
        try:
            resultados.append(vectorize_webmap(wm["id"], deep=deep))
        except Exception as e:
            logger.error(f"Error vectorizando {wm['titulo']}: {e}")
            errores.append({"webmap_id": wm["id"], "titulo": wm["titulo"], "error": str(e)})

    return {
        "ok"          : len(errores) == 0,
        "vectorizados": len(resultados),
        "fallidos"    : len(errores),
        "detalle"     : resultados,
        "errores"     : errores,
    }


def search_webmap(query: str, webmap_id: str = None, k: int = 5) -> List[dict]:
    """Busca por similitud dentro de un WebMap, o en el catálogo global."""
    emb = get_embeddings()
    vector = emb.embed_query(query)

    name = collection_name(webmap_id) if webmap_id else settings.chroma_catalog
    if webmap_id == "__layers__":
        name = settings.chroma_layers
    try:
        col = get_chroma_client().get_collection(name)
    except Exception:
        return []

    hecho_con = (col.metadata or {}).get("embedding")
    actual = f"{emb.provider}/{emb.model}"
    if hecho_con != actual:
        raise ValueError(
            f"El índice se creó con embeddings '{hecho_con or 'desconocidos'}' y ahora se usan "
            f"'{actual}'. Re-vectorizá el mapa para regenerarlo."
        )

    n = col.count()
    if n == 0:
        return []
    res = col.query(query_embeddings=[vector], n_results=min(k, n))

    hits = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({
            "texto"    : doc,
            "metadata" : meta or {},
            "similitud": round(1 - float(dist), 4),
        })
    return hits


_SISTEMA = {"objectid", "fid", "globalid", "creator", "editor", "creationdate", "editdate",
            "shapearea", "shapelength", "shape"}


def compact_fields(campos: str, max_chars: int = 400) -> str:
    """
    'nombre (alias), …' del documento indexado → lista compacta para el prompt.
    Quita los campos de sistema y recorta los alias largos (los de Survey123 son la
    pregunta entera). Sin esto, los campos que van al final (p. ej. intencion_bert)
    quedaban fuera del recorte y el modelo no sabía que existían.
    """
    # Separar por comas de primer nivel: un alias puede traer paréntesis, "Intención (BERT)"
    trozos, actual, nivel = [], "", 0
    for ch in campos or "":
        nivel += (ch == "(") - (ch == ")")
        if ch == "," and nivel == 0:
            trozos.append(actual)
            actual = ""
        else:
            actual += ch
    trozos.append(actual)

    partes = []
    for t in (t.strip() for t in trozos if t.strip()):
        nombre, _, alias = t.partition("(")
        alias = alias.strip()
        nombre, alias = nombre.strip(), (alias[:-1] if alias.endswith(")") else alias).strip()
        if not nombre:
            continue
        if re.sub(r"[^a-z]", "", nombre.lower()) in _SISTEMA:
            continue
        if not alias or alias == nombre or alias.startswith("esriFieldType"):
            partes.append(nombre)
        else:
            partes.append(f"{nombre} ({alias[:28] + '…' if len(alias) > 30 else alias})")
    out = ", ".join(partes) if partes else (campos or "")
    return out if len(out) <= max_chars else out[:max_chars].rsplit(",", 1)[0] + ", …"


def webmap_layers_brief(webmap_id: str, max_layers: int = 15, max_campos: int = 400) -> str:
    """
    Lista compacta de las capas de un WebMap vectorizado: título, geometría,
    registros, URL y campos. Se lee directo de la colección (col.get), así que
    no cuesta embeddings ni llamadas al LLM. La usa el Planner para saber qué
    hay en el mapa abierto antes de decidir qué agente usar.
    """
    try:
        col = get_chroma_client().get_collection(collection_name(webmap_id))
        data = col.get(include=["documents", "metadatas"])
    except Exception:
        return ""
    titulo, capas = "", []
    for doc, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
        meta = meta or {}
        titulo = titulo or meta.get("webmap_titulo", "")
        if meta.get("tipo") != "capa":
            continue
        campos = doc.split("Campos:", 1)[1].strip() if "Campos:" in doc else ""
        m = re.search(r"^Valores: (.*)$", doc, re.MULTILINE)
        valores = f"\n  valores: {m.group(1)}" if m else ""
        geom = (meta.get("geometria") or "").replace("esriGeometry", "").lower() or "n/d"
        url = meta.get("capa_url") or "(sin URL: capa de grupo o no consultable)"
        capas.append(
            f"- {meta.get('capa_titulo')} · geometría {geom} · "
            f"{meta.get('total_features') or 'n/d'} registros\n"
            f"  url: {url}\n  campos: {compact_fields(campos, max_campos) or 'n/d'}{valores}"
        )
    if not capas:
        return ""
    extra = f"\n(… y {len(capas) - max_layers} capas más)" if len(capas) > max_layers else ""
    return f"WebMap «{titulo}» con {len(capas)} capas:\n" + "\n".join(capas[:max_layers]) + extra


def vector_status() -> dict:
    """Estado del índice vectorial — lo consume /api/vector-status."""
    try:
        client = get_chroma_client()
        cols = client.list_collections()
        names = [c.name if hasattr(c, "name") else str(c) for c in cols]

        webmaps = []
        try:
            catalog = client.get_collection(settings.chroma_catalog)
            data = catalog.get()
            for meta in data.get("metadatas", []) or []:
                webmaps.append(meta)
        except Exception:
            pass

        total = 0
        for n in names:
            try:
                total += client.get_collection(n).count()
            except Exception:
                continue

        return {
            "ok"          : True,
            "modo"        : "http" if settings.chroma_host else "local",
            "ubicacion"   : (
                f"{settings.chroma_host}:{settings.chroma_port}"
                if settings.chroma_host else settings.chroma_path
            ),
            "colecciones" : len(names),
            "documentos"  : total,
            "webmaps"     : webmaps,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def delete_webmap_index(webmap_id: str) -> dict:
    """Borra el índice de un WebMap para poder re-vectorizarlo desde cero."""
    client = get_chroma_client()
    name = collection_name(webmap_id)
    try:
        client.delete_collection(name)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    try:
        client.get_collection(settings.chroma_catalog).delete(ids=[webmap_id])
    except Exception:
        pass
    try:
        client.get_collection(settings.chroma_layers).delete(where={"webmap_id": webmap_id})
    except Exception:
        pass
    return {"ok": True, "eliminado": name}


# ══════════════════════════════════════════════════════════
# TOOLS DE LANGCHAIN (las usa el RAGAgent)
# ══════════════════════════════════════════════════════════

@tool
def search_webmap_context(input_str: str) -> str:
    """
    Busca información sobre las capas de un WebMap ya vectorizado.
    Action Input: JSON con query y opcionalmente webmap_id y k.
    Ejemplo: {"query": "capas de arbolado urbano", "webmap_id": "a1b2c3", "k": 5}
    """
    try:
        params = input_str if isinstance(input_str, dict) else json.loads(str(input_str))
        hits = search_webmap(
            query=params.get("query", ""),
            webmap_id=params.get("webmap_id") or None,
            k=int(params.get("k", 5)),
        )
        return json.dumps({"total": len(hits), "resultados": hits}, ensure_ascii=False)
    except json.JSONDecodeError:
        hits = search_webmap(query=str(input_str), k=5)
        return json.dumps({"total": len(hits), "resultados": hits}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "mensaje": str(e)}, ensure_ascii=False)


@tool
def list_vectorized_webmaps(dummy: str = "") -> str:
    """
    Lista los WebMaps que ya están vectorizados en ChromaDB.
    Action Input: dejar vacío.
    """
    try:
        st = vector_status()
        return json.dumps({
            "total"  : len(st.get("webmaps", [])),
            "webmaps": st.get("webmaps", []),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "mensaje": str(e)}, ensure_ascii=False)


@tool
def get_webmap_layers(webmap_id: str) -> str:
    """
    Devuelve las capas de un WebMap directamente desde ArcGIS Online.
    Action Input: el ID del WebMap, ejemplo: a1b2c3d4e5f6
    """
    try:
        wm = read_webmap(str(webmap_id).strip().strip('"').strip("'"))
        return json.dumps({
            "webmap": wm["titulo"],
            "total" : wm["total_capas"],
            "capas" : [
                {"titulo": c["titulo"], "tipo": c["tipo"], "url": c["url"]}
                for c in wm["capas"]
            ],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "mensaje": str(e)}, ensure_ascii=False)


VECTOR_TOOLS = [search_webmap_context, list_vectorized_webmaps, get_webmap_layers]
