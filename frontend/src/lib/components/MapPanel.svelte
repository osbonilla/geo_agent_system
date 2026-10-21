<script lang="ts">
  /**
   * Visor. Arranca con un mapa base sobre Quito y lo reemplaza por el
   * WebMap que se elija. Los agentes le mandan capas y features.
   */
  import { onMount } from 'svelte';
  import { esri, geojsonLayer } from '../arcgis';

  let { webmapId = null as string | null } = $props();

  let container: HTMLDivElement;
  let view: any = null;
  let resultLayer: any = null;
  let agentLayers: any[] = [];

  let estado = $state('cargando el SDK de ArcGIS');
  let error = $state('');
  let coords = $state('—');
  let escala = $state('');
  let capasAgente = $state<string[]>([]);

  const QUITO = { center: [-78.4678, -0.1807], zoom: 11 };

  onMount(async () => {
    try {
      const [Map, MapView] = await esri<[any, any]>([
        'esri/Map',
        'esri/views/MapView',
      ]);

      view = new MapView({
        container,
        map: new Map({
          basemap: document.body.classList.contains('calcite-mode-light') ? 'gray-vector' : 'dark-gray-vector',
        }),
        center: QUITO.center,
        zoom: QUITO.zoom,
        ui: { components: ['zoom', 'attribution'] },
      });

      await view.when();
      estado = 'mapa base';

      view.on('pointer-move', (e: any) => {
        const p = view.toMap({ x: e.x, y: e.y });
        if (p) coords = `${p.latitude.toFixed(4)}, ${p.longitude.toFixed(4)}`;
      });

      view.watch('scale', (s: number) => {
        escala = `1:${Math.round(s).toLocaleString('es-EC')}`;
      });
    } catch (e: any) {
      error = e.message ?? String(e);
      estado = 'sin visor';
    }
  });

  // Cambiar de WebMap: se reemplaza el mapa entero, no las capas
  $effect(() => {
    const id = webmapId;
    if (!id || !view) return;

    (async () => {
      estado = 'abriendo WebMap';
      try {
        const [WebMap] = await esri<[any]>(['esri/WebMap']);
        const wm = new WebMap({ portalItem: { id } });
        await wm.load();
        view.map = wm;
        resultLayer = null;
        agentLayers = [];
        capasAgente = [];
        estado = `WebMap · ${wm.portalItem?.title ?? id}`;
      } catch (e: any) {
        error = `No se pudo abrir el WebMap: ${e.message ?? e}`;
        estado = 'error al abrir';
      }
    })();
  });

  export async function drawGeoJSON(fc: GeoJSON.FeatureCollection, title = 'Respuesta del agente') {
    if (!view) return;
    if (resultLayer) view.map.remove(resultLayer);
    resultLayer = await geojsonLayer(fc, title);
    view.map.add(resultLayer);
    await resultLayer.when();
    const extent = await resultLayer.queryExtent();
    if (extent?.extent) view.goTo(extent.extent.expand(1.4));
  }

  export async function addFeatureLayer(url: string, title: string) {
    if (!view || agentLayers.some((l) => l.url === url)) return;
    const [FeatureLayer] = await esri<[any]>(['esri/layers/FeatureLayer']);
    const layer = new FeatureLayer({ url, title, opacity: 0.9 });
    view.map.add(layer);
    agentLayers = [...agentLayers, layer];
    capasAgente = [...capasAgente, title];
    try {
      await layer.when();
      if (layer.fullExtent) view.goTo(layer.fullExtent);
    } catch { /* la capa puede no tener extent publicado */ }
  }

  export function clearAgentLayers() {
    if (!view) return;
    for (const l of agentLayers) view.map.remove(l);
    if (resultLayer) view.map.remove(resultLayer);
    agentLayers = [];
    resultLayer = null;
    capasAgente = [];
  }
</script>

<div class="visor">
  <div class="canvas" bind:this={container}></div>

  {#if error}
    <calcite-notice class="alerta" open kind="danger" icon="exclamation-mark-triangle" closable
      oncalciteNoticeClose={() => (error = '')}>
      <div slot="title">Visor</div>
      <div slot="message">{error}</div>
    </calcite-notice>
  {/if}

  <footer class="regleta">
    <span class="estado">
      <calcite-icon icon="map" scale="s"></calcite-icon>
      {estado}
    </span>
    <div class="lecturas">
      {#if capasAgente.length}
        <calcite-chip scale="s" kind="brand" icon="layers" closable label="Capas del agente"
          oncalciteChipClose={clearAgentLayers}>
          {capasAgente.length} capa{capasAgente.length > 1 ? 's' : ''} del agente
        </calcite-chip>
      {/if}
      <span class="mono">{escala}</span>
      <span class="mono">{coords}</span>
    </div>
  </footer>
</div>

<style>
  .visor {
    position: relative;
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    min-height: 320px;
  }

  .canvas { flex: 1; min-height: 0; }

  .regleta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 6px 12px;
    border-top: 1px solid var(--calcite-color-border-3);
    background: var(--calcite-color-foreground-1);
    font-size: var(--calcite-font-size--2, 11px);
    color: var(--calcite-color-text-3);
  }

  .estado { display: flex; align-items: center; gap: 6px; }

  .lecturas { display: flex; align-items: center; gap: 14px; }

  .alerta {
    position: absolute;
    inset: auto 12px 48px 12px;
    z-index: 2;
  }
</style>
