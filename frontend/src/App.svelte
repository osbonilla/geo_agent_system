<script lang="ts">
  /**
   * Armazón Calcite:
   *   calcite-shell
   *   ├── header       → InstrumentStrip (estado del sistema + modelo)
   *   ├── panel-start  → WebMapPicker (mapas de la organización)
   *   ├── centro       → MapPanel (visor ArcGIS) — el protagonista
   *   └── panel-end    → ChatPanel (consulta a los agentes)
   *
   * Los dos paneles se pliegan desde su barra de acciones (quedan como una
   * franja de iconos) para dejarle el espacio al mapa. El estado plegado se
   * recuerda en el navegador.
   */
  import InstrumentStrip from './lib/components/InstrumentStrip.svelte';
  import WebMapPicker from './lib/components/WebMapPicker.svelte';
  import MapPanel from './lib/components/MapPanel.svelte';
  import ChatPanel from './lib/components/ChatPanel.svelte';
  import type { ChatReply } from './lib/api';
  import { aplicarModo, modoInicial, type Modo } from './lib/theme';

  let webmapId = $state<string | null>(null);
  let webmapTitulo = $state('');
  const inicial = modoInicial();
  let modo = $state<Modo>(inicial);
  aplicarModo(inicial);

  let mapa: MapPanel;
  let chat: ChatPanel;

  function leer(clave: string, porDefecto: boolean): boolean {
    try {
      const v = localStorage.getItem(clave);
      return v === null ? porDefecto : v === '1';
    } catch {
      return porDefecto;
    }
  }
  function guardar(clave: string, v: boolean) {
    try { localStorage.setItem(clave, v ? '1' : '0'); } catch { /* sin almacenamiento */ }
  }

  let mapasPlegado = $state(leer('geoagent.mapasPlegado', false));
  let chatPlegado = $state(leer('geoagent.chatPlegado', false));
  let respuestaNueva = $state(false);   // aviso en el icono del chat si llega una respuesta plegado

  function alternarMapas() {
    mapasPlegado = !mapasPlegado;
    guardar('geoagent.mapasPlegado', mapasPlegado);
  }
  function alternarChat() {
    chatPlegado = !chatPlegado;
    if (!chatPlegado) respuestaNueva = false;
    guardar('geoagent.chatPlegado', chatPlegado);
  }

  function elegirMapa(id: string | null, titulo: string) {
    webmapId = id;
    webmapTitulo = titulo;
  }

  function cambiarModo() {
    modo = modo === 'dark' ? 'light' : 'dark';
    aplicarModo(modo);
  }

  async function alResponder(r: ChatReply) {
    if (chatPlegado) respuestaNueva = true;
    if (r.geojson?.features?.length) {
      await mapa.drawGeoJSON(r.geojson, `Respuesta · ${r.agent_used}`);
    } else if (r.map_action?.type === 'add_layer' && r.map_action.url) {
      await mapa.addFeatureLayer(r.map_action.url, r.map_action.title ?? 'Capa citada');
    }
  }
</script>

<calcite-shell>
  <div slot="header">
    <InstrumentStrip {modo} onToggleModo={cambiarModo} />
  </div>

  <calcite-shell-panel slot="panel-start" width="s" resizable collapsed={mapasPlegado}>
    <calcite-action-bar slot="action-bar" expand-disabled>
      <calcite-action
        icon={mapasPlegado ? 'chevrons-right' : 'chevrons-left'}
        text={mapasPlegado ? 'Mostrar mapas web' : 'Ocultar mapas web'}
        label="Mostrar u ocultar la lista de mapas web"
        onclick={alternarMapas}
      ></calcite-action>
      <calcite-action icon="map" text="Mapas web" active={!mapasPlegado}
        onclick={() => mapasPlegado && alternarMapas()}></calcite-action>
    </calcite-action-bar>
    <WebMapPicker
      selected={webmapId}
      onSelect={elegirMapa}
      onVectorized={(m: string, ok: boolean) => chat.anunciar(m, ok)}
    />
  </calcite-shell-panel>

  <MapPanel bind:this={mapa} {webmapId} />

  <calcite-shell-panel slot="panel-end" width="m" resizable collapsed={chatPlegado}>
    <calcite-action-bar slot="action-bar" expand-disabled>
      <calcite-action
        icon={chatPlegado ? 'chevrons-left' : 'chevrons-right'}
        text={chatPlegado ? 'Mostrar consola' : 'Ocultar consola'}
        label="Mostrar u ocultar la consola de agentes"
        onclick={alternarChat}
      ></calcite-action>
      <calcite-action icon="speech-bubbles" text="Consola de agentes" active={!chatPlegado}
        indicator={respuestaNueva} onclick={() => chatPlegado && alternarChat()}></calcite-action>
    </calcite-action-bar>
    <ChatPanel
      bind:this={chat}
      {webmapId}
      {webmapTitulo}
      onReply={alResponder}
      onLayer={(url: string, titulo: string) => mapa.addFeatureLayer(url, titulo)}
    />
  </calcite-shell-panel>
</calcite-shell>
