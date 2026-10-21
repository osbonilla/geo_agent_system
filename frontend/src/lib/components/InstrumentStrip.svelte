<script lang="ts">
  /**
   * Barra superior (calcite-navigation): estado de las cuatro piezas del
   * sistema y selector de modelo. Si falta una API key, el mensaje del
   * router aparece tal cual, con el enlace para conseguirla.
   */
  import { api, type Status, type Provider } from '../api';
  import type { Modo } from '../theme';

  let { modo = 'dark' as Modo, onToggleModo = () => {} } = $props();

  let estado = $state<Status | null>(null);
  let proveedores = $state<Provider[]>([]);
  let abierto = $state(false);
  let modelo = $state('');
  let cambiando = $state(false);
  let aviso = $state('');

  // Se reemplaza por los modelos instalados en Ollama al abrir el selector
  let SLM = $state(['ollama/qwen2.5:7b', 'ollama/mistral:7b']);

  async function refrescar() {
    try {
      estado = await api.status();
      if (!abierto) modelo = estado.llm.full ?? modelo;
    } catch (e: any) {
      aviso = e.message;
    }
  }

  async function abrirSelector() {
    abierto = true;
    aviso = '';
    if (!proveedores.length) {
      try {
        proveedores = (await api.providers()).providers;
      } catch { /* la lista es opcional */ }
    }
    try {
      const instalados = (await api.ollamaModels()).models;
      if (instalados.length) SLM = instalados;
    } catch { /* si Ollama no responde, quedan los de por defecto */ }
  }

  async function aplicar() {
    if (!modelo.trim()) return;
    cambiando = true;
    aviso = '';
    try {
      await api.selectModel(modelo.trim());
      await refrescar();
      abierto = false;
    } catch (e: any) {
      aviso = e.message;
    } finally {
      cambiando = false;
    }
  }

  const ok = (v?: boolean) => (v ? 'check-circle' : 'exclamation-mark-triangle');
  const tono = (v?: boolean) =>
    `--calcite-chip-icon-color: var(${v ? '--calcite-color-status-success' : '--calcite-color-status-danger'})`;

  refrescar();
  setInterval(refrescar, 30000);
</script>

<calcite-navigation>
  <calcite-navigation-logo
    slot="logo"
    heading="Agente Geoespacial"
    description="Arquitectura multi-agente · ArcGIS Online"
    icon="globe"
  ></calcite-navigation-logo>

  <div slot="content-end" class="estado">
    <calcite-button
      appearance="outline-fill"
      kind={estado && !estado.llm.ok ? 'danger' : 'neutral'}
      icon-start={estado?.llm.local ? 'desktop' : 'globe'}
      scale="s"
      title="Cambiar de modelo"
      onclick={abrirSelector}
    >
      <span class="mono">{estado?.llm.ok ? estado.llm.full : 'LLM sin configurar'}</span>
    </calcite-button>

    <calcite-chip scale="s" appearance="outline-fill" kind="neutral" icon={ok(estado?.arcgis.ok)}
      style={tono(estado?.arcgis.ok)} label="ArcGIS"
      title={estado?.arcgis.error ?? ''}>
      ArcGIS · {estado?.arcgis.usuario ?? 'sin conexión'}
    </calcite-chip>

    <calcite-chip scale="s" appearance="outline-fill" kind="neutral" icon={ok(estado?.database.ok)}
      style={tono(estado?.database.ok)} label="PostgreSQL"
      title={estado?.database.error ?? ''}>
      PostGIS
    </calcite-chip>

    <calcite-chip scale="s" appearance="outline-fill" kind="neutral" icon={ok(estado?.vectors.ok)}
      style={tono(estado?.vectors.ok)} label="Índice vectorial"
      title={estado?.vectors.error ?? ''}>
      Índice · {estado?.vectors.ok ? `${estado.vectors.documentos ?? 0} docs` : 'sin chroma'}
    </calcite-chip>

    <calcite-action
      text={modo === 'dark' ? 'Modo claro' : 'Modo oscuro'}
      icon={modo === 'dark' ? 'brightness' : 'moon'}
      scale="s"
      onclick={onToggleModo}
    ></calcite-action>
  </div>
</calcite-navigation>

<calcite-dialog
  open={abierto}
  modal
  heading="Modelo de lenguaje"
  description="SLM local (Ollama) o LLM por API, con el formato proveedor/modelo"
  width-scale="m"
  oncalciteDialogClose={() => (abierto = false)}
>
  {#if estado && !estado.llm.ok}
    <calcite-notice open kind="danger" icon="exclamation-mark-triangle" width="full">
      <div slot="message">{estado.llm.error}</div>
    </calcite-notice>
  {/if}
  {#if aviso}
    <calcite-notice open kind="danger" icon="exclamation-mark-triangle" width="full">
      <div slot="message">{aviso}</div>
    </calcite-notice>
  {/if}

  <calcite-label>
    Modelo activo
    <calcite-input-text
      value={modelo}
      placeholder="ollama/qwen2.5:7b"
      oncalciteInputTextInput={(e: any) => (modelo = e.target.value)}
      onkeydown={(e: KeyboardEvent) => e.key === 'Enter' && aplicar()}
    ></calcite-input-text>
  </calcite-label>

  <calcite-label>
    SLM instalados en Ollama
    <div class="opciones">
      {#each SLM as m}
        <calcite-chip scale="s" kind={modelo === m ? 'brand' : 'neutral'} appearance="outline-fill"
          label={m} onclick={() => (modelo = m)} class="clic">{m}</calcite-chip>
      {/each}
    </div>
  </calcite-label>

  <calcite-list label="Proveedores" selection-mode="none">
    {#each proveedores as p}
      <calcite-list-item
        label={p.label}
        description={p.example}
        onclick={() => (modelo = p.example)}
      >
        <calcite-icon slot="content-start" icon={p.local ? 'desktop' : 'globe'} scale="s"></calcite-icon>
        {#if p.ready}
          <calcite-chip slot="content-end" scale="s" kind="brand" label="listo">listo</calcite-chip>
        {:else}
          <calcite-link slot="content-end" href={p.signup_url} target="_blank" rel="noreferrer">
            falta {p.env_key}
          </calcite-link>
        {/if}
      </calcite-list-item>
    {/each}
  </calcite-list>

  <p class="nota">
    El cambio dura mientras el servidor siga corriendo. Para dejarlo fijo,
    edita <span class="mono">LLM_MODEL</span> en el <span class="mono">.env</span>.
  </p>

  <calcite-button slot="footer-start" appearance="transparent" kind="neutral"
    onclick={() => (abierto = false)}>Cancelar</calcite-button>
  <calcite-button slot="footer-end" icon-start="check" loading={cambiando}
    disabled={cambiando} onclick={aplicar}>Usar este modelo</calcite-button>
</calcite-dialog>

<style>
  .estado {
    display: flex;
    align-items: center;
    gap: var(--calcite-spacing-sm, 8px);
    padding-inline: var(--calcite-spacing-md, 12px);
    flex-wrap: wrap;
  }

  .opciones {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
  }

  .clic { cursor: pointer; }

  .nota {
    margin: 12px 0 0;
    font-size: var(--calcite-font-size--1, 12px);
    color: var(--calcite-color-text-3);
  }

  calcite-notice { margin-bottom: 12px; }

  @media (max-width: 900px) {
    calcite-chip { display: none; }
  }
</style>
