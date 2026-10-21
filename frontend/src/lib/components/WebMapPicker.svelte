<script lang="ts">
  /**
   * Selector de WebMaps (calcite-panel + calcite-list). Cada mapa muestra si
   * ya está vectorizado; la acción de la derecha lo indexa en ChromaDB.
   */
  import { api, type WebMap } from '../api';

  let {
    selected = null as string | null,
    onSelect = (_id: string | null, _titulo: string) => {},
    onVectorized = (_msg: string, _ok: boolean) => {},
  } = $props();

  let mapas = $state<WebMap[]>([]);
  let cargando = $state(true);
  let error = $state('');
  let trabajando = $state<string | null>(null);

  async function cargar() {
    cargando = true;
    error = '';
    try {
      mapas = (await api.webmaps()).webmaps;
    } catch (e: any) {
      error = e.message;
    } finally {
      cargando = false;
    }
  }

  async function vectorizar(m: WebMap, ev: Event) {
    ev.stopPropagation();
    trabajando = m.id;
    try {
      const r = await api.vectorize(m.id, true);
      mapas = mapas.map((x) =>
        x.id === m.id ? { ...x, vectorizado: true, embedding: r.embedding } : x,
      );
      onVectorized(
        `«${r.titulo}» quedó indexado: ${r.capas} capas, ${r.documentos} documentos (embeddings ${r.embedding}).`,
        true,
      );
    } catch (e: any) {
      onVectorized(`No se pudo vectorizar «${m.titulo}»: ${e.message}`, false);
    } finally {
      trabajando = null;
    }
  }

  function alCambiar(e: any) {
    const item = e.target.selectedItems?.[0];
    const m = mapas.find((x) => x.id === item?.value);
    onSelect(m ? m.id : null, m ? m.titulo : '');
  }

  cargar();
</script>

<calcite-panel heading="Mapas web" description="ArcGIS Online">
  <calcite-action
    slot="header-actions-end"
    icon="refresh"
    text="Actualizar"
    loading={cargando}
    onclick={cargar}
  ></calcite-action>

  {#if error}
    <calcite-notice open kind="danger" icon="exclamation-mark-triangle" width="full">
      <div slot="title">No se pudo leer ArcGIS Online</div>
      <div slot="message">{error}</div>
      <div slot="message" class="pista">Revisá ARCGIS_USERNAME y ARCGIS_PASSWORD en el .env.</div>
    </calcite-notice>
  {:else if cargando && !mapas.length}
    <calcite-loader label="Consultando la organización" text="Consultando la organización…"></calcite-loader>
  {:else if !mapas.length}
    <calcite-notice open icon="information" width="full">
      <div slot="message">No hay WebMaps para mostrar.</div>
    </calcite-notice>
  {:else}
    <calcite-list
      label="WebMaps"
      selection-mode="single"
      selection-appearance="border"
      filter-enabled={mapas.length > 4}
      filter-placeholder="Filtrar por nombre o etiqueta"
      oncalciteListChange={alCambiar}
    >
      {#each mapas as m (m.id)}
        <calcite-list-item
          value={m.id}
          label={m.titulo}
          description={m.snippet || m.tags.join(', ')}
          selected={selected === m.id}
        >
          <div slot="content-bottom" class="etiqueta">
          <calcite-chip
            scale="s"
            appearance="outline-fill"
            kind={m.vectorizado ? 'brand' : 'neutral'}
            icon={m.vectorizado ? 'check' : 'circle-disallowed'}
            label={m.vectorizado ? 'Indexado' : 'Sin índice'}
          >
            {m.vectorizado ? 'Indexado' : 'Sin índice'}
          </calcite-chip>
          </div>
          <calcite-action
            slot="actions-end"
            icon={m.vectorizado ? 'refresh' : 'plus'}
            text={m.vectorizado ? 'Re-vectorizar' : 'Vectorizar'}
            title={m.vectorizado ? 'Re-vectorizar' : 'Vectorizar'}
            loading={trabajando === m.id}
            disabled={trabajando !== null}
            onclick={(e: Event) => vectorizar(m, e)}
          ></calcite-action>
        </calcite-list-item>
      {/each}
    </calcite-list>
  {/if}

  <div slot="footer" class="pie">
    {selected ? 'Las preguntas usan las capas de este mapa: el Planner decide si describirlo o consultar sus datos.' : 'Sin mapa elegido: el Planner busca en toda tu organización.'}
  </div>
</calcite-panel>

<style>
  .pie {
    font-size: var(--calcite-font-size--2, 11px);
    color: var(--calcite-color-text-3);
  }

  .etiqueta { padding: 0 12px 8px; }

  .pista { margin-top: 4px; font-size: var(--calcite-font-size--1, 12px); }
</style>
