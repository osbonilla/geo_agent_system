<script lang="ts">
  /**
   * Consola de chat (calcite-panel). Cada respuesta muestra el agente que la
   * produjo, sus métricas (tiempo, llamadas al LLM, tokens), las capas que
   * citó y los pasos que ejecutó.
   */
  import { api, type ChatReply, type Source } from '../api';
  import { markdown } from '../markdown';

  type Turno = {
    rol: 'usuario' | 'agente' | 'sistema';
    texto: string;
    agente?: string;
    ms?: number;
    llamadas?: number;
    tokens?: number;
    plan?: { agente: string; tarea: string }[];
    pasos?: { tool: string; input: string; output: string }[];
    fuentes?: Source[];
    error?: boolean;
  };

  let {
    webmapId = null as string | null,
    webmapTitulo = '',
    onReply = (_r: ChatReply) => {},
    onLayer = (_url: string, _titulo: string) => {},
  } = $props();

  let turnos = $state<Turno[]>([]);
  let borrador = $state('');
  let pensando = $state(false);
  let transcripcion: HTMLDivElement;

  const sugerencias = $derived(
    webmapId
      ? [
          '¿Qué capas tiene este mapa y de qué tratan?',
          '¿Cuántos registros tiene cada capa de este mapa?',
          'Resume la capa principal de este mapa con sus estadísticas',
          '¿Qué parroquia concentra más registros de la capa principal?',
        ]
      : [
          '¿Qué parroquia tiene más denuncias ciudadanas?',
          '¿Qué parroquias del DMQ están en zona de amenaza volcánica?',
          '¿Cuántas áreas protegidas hay a menos de 50 km de Quito?',
        ],
  );

  export function anunciar(texto: string, ok = true) {
    turnos = [...turnos, { rol: 'sistema', texto, error: !ok }];
    bajar();
  }

  function bajar() {
    queueMicrotask(() => {
      transcripcion?.scrollTo({ top: transcripcion.scrollHeight, behavior: 'smooth' });
    });
  }

  async function enviar() {
    const texto = borrador.trim();
    if (!texto || pensando) return;

    turnos = [...turnos, { rol: 'usuario', texto }];
    borrador = '';
    pensando = true;
    bajar();

    try {
      const r = await api.chat(texto, webmapId);
      turnos = [
        ...turnos,
        {
          rol: 'agente',
          texto: r.answer,
          agente: r.agent_used,
          ms: r.duration_ms,
          llamadas: r.metrics?.llm_calls,
          tokens: (r.metrics?.prompt_tokens ?? 0) + (r.metrics?.completion_tokens ?? 0),
          plan: r.plan,
          pasos: r.steps,
          fuentes: r.sources,
          error: !r.success,
        },
      ];
      onReply(r);
    } catch (e: any) {
      turnos = [...turnos, { rol: 'agente', texto: e.message, agente: 'Error', error: true }];
    } finally {
      pensando = false;
      bajar();
    }
  }

  function teclas(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      enviar();
    }
  }

  function metricas(t: Turno) {
    const partes = [];
    if (t.ms) partes.push(`${(t.ms / 1000).toFixed(1)} s`);
    if (t.llamadas) partes.push(`${t.llamadas} llamada${t.llamadas > 1 ? 's' : ''} LLM`);
    if (t.tokens) partes.push(`${t.tokens.toLocaleString('es-EC')} tokens`);
    return partes.join(' · ');
  }
</script>

<calcite-panel
  heading={webmapId ? webmapTitulo : 'Toda la organización'}
  description={webmapId ? 'Planner · sobre las capas de este mapa' : 'Planner · reparte entre los agentes'}
>
  <calcite-action
    slot="header-actions-end"
    icon="trash"
    text="Limpiar conversación"
    disabled={!turnos.length}
    onclick={() => (turnos = [])}
  ></calcite-action>

  <div class="transcripcion" bind:this={transcripcion}>
    {#if !turnos.length}
      <div class="intro">
        <p>
          Con un mapa elegido, el Planner conoce sus capas: describe el mapa (RAGAgent)
          consulta los datos de una capa (AnalyticsAgent, GeoAgent) o cruza capas por su
          ubicación (SpatialAgent). Sin mapa, busca
          en toda tu organización de ArcGIS Online.
        </p>
        <div class="sugerencias">
          {#each sugerencias as s}
            <calcite-button appearance="outline" kind="neutral" scale="s" width="full"
              alignment="start" icon-start="lightbulb" onclick={() => (borrador = s)}>
              {s}
            </calcite-button>
          {/each}
        </div>
        <calcite-block heading="Prefijos" description="Para mandar la pregunta a un agente concreto" collapsible>
          <dl class="prefijos">
            <dt class="mono">geo:</dt><dd>capas y registros en vivo desde ArcGIS Online</dd>
            <dt class="mono">analytics:</dt><dd>estadísticas sobre un Feature Layer</dd>
            <dt class="mono">postgis:</dt><dd>consultas espaciales sobre lo ya guardado</dd>
            <dt class="mono">rag:</dt><dd>solo el índice vectorial, sin consultar datos</dd>
            <dt class="mono">plan:</dt><dd>Plan &amp; Execute con LangGraph</dd>
          </dl>
        </calcite-block>
      </div>
    {/if}

    {#each turnos as t, i (i)}
      {#if t.rol === 'usuario'}
        <div class="burbuja usuario">{t.texto}</div>
      {:else if t.rol === 'sistema'}
        <calcite-notice open width="full" scale="s"
          kind={t.error ? 'danger' : 'success'}
          icon={t.error ? 'exclamation-mark-triangle' : 'check-circle'}>
          <div slot="title">Índice vectorial</div>
          <div slot="message">{t.texto}</div>
        </calcite-notice>
      {:else}
        <article class="burbuja agente" class:falla={t.error}>
          <header>
            <calcite-chip scale="s" kind={t.error ? 'neutral' : 'brand'}
              icon={t.error ? 'exclamation-mark-triangle' : 'analysis'} label={t.agente ?? ''}>
              {t.agente}
            </calcite-chip>
            <span class="metricas mono">{metricas(t)}</span>
          </header>

          <div class="md">{@html markdown(t.texto)}</div>

          {#if t.fuentes?.length}
            <calcite-list label="Capas citadas" scale="s" selection-mode="none">
              {#each t.fuentes as f}
                <calcite-list-item label={f.capa} description={`${f.registros || '—'} registros`}>
                  <calcite-action slot="actions-end" icon="add-layer" text="Agregar al mapa"
                    title="Agregar al mapa" onclick={() => onLayer(f.url, f.capa)}></calcite-action>
                </calcite-list-item>
              {/each}
            </calcite-list>
          {/if}

          {#if t.plan?.length || t.pasos?.length}
            <calcite-block
              heading={`${t.pasos?.length ?? 0} paso${(t.pasos?.length ?? 0) === 1 ? '' : 's'}`}
              description={t.plan?.length ? `Plan: ${t.plan.map((p) => p.agente).join(' → ')}` : ''}
              collapsible
              scale="s"
            >
              {#each t.pasos ?? [] as p}
                <div class="paso mono">
                  <strong>{p.tool}</strong>
                  <span class="entrada">{String(p.input).slice(0, 160)}</span>
                  <span class="salida">{String(p.output).slice(0, 260)}</span>
                </div>
              {/each}
            </calcite-block>
          {/if}
        </article>
      {/if}
    {/each}

    {#if pensando}
      <div class="pensando">
        <calcite-loader inline scale="s" label="Consultando agentes"></calcite-loader>
        <span>Consultando agentes…</span>
      </div>
    {/if}
  </div>

  <div slot="footer" class="compositor">
    <calcite-text-area
      value={borrador}
      rows="3"
      resize="vertical"
      label="Pregunta"
      placeholder={webmapId
        ? `Pregunta sobre «${webmapTitulo}»`
        : 'Pregunta sobre tus capas o usa un prefijo: geo: analytics: spatial: rag: plan:'}
      oncalciteTextAreaInput={(e: any) => (borrador = e.target.value)}
      onkeydown={teclas}
    ></calcite-text-area>
    <div class="acciones">
      <span class="ayuda">Enter envía · Shift+Enter nueva línea</span>
      <calcite-button scale="s" icon-start="send" loading={pensando}
        disabled={pensando || !borrador.trim()} onclick={enviar}>Enviar</calcite-button>
    </div>
  </div>
</calcite-panel>

<style>
  .transcripcion {
    display: flex;
    flex-direction: column;
    gap: var(--calcite-spacing-md, 12px);
    padding: var(--calcite-spacing-md, 12px);
  }

  .intro p {
    margin: 0 0 12px;
    color: var(--calcite-color-text-2);
    font-size: var(--calcite-font-size--1, 13px);
    line-height: 1.5;
  }

  .sugerencias {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 12px;
  }

  .prefijos {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    margin: 0;
    font-size: var(--calcite-font-size--1, 13px);
  }

  .prefijos dt { color: var(--calcite-color-brand); }
  .prefijos dd { margin: 0; color: var(--calcite-color-text-2); }

  .burbuja {
    border-radius: var(--calcite-corner-radius-round, 4px);
    font-size: var(--calcite-font-size-0, 14px);
    line-height: 1.5;
  }

  .burbuja.usuario {
    align-self: flex-end;
    max-width: 85%;
    padding: 8px 12px;
    background: var(--calcite-color-brand);
    color: var(--calcite-color-text-inverse);
    white-space: pre-wrap;
  }

  .burbuja.agente {
    padding: 10px 12px;
    background: var(--calcite-color-foreground-1);
    border: 1px solid var(--calcite-color-border-3);
    border-inline-start: 3px solid var(--calcite-color-brand);
  }

  .burbuja.agente.falla { border-inline-start-color: var(--calcite-color-status-danger); }

  .burbuja header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }

  .md { margin: 8px 0; color: var(--calcite-color-text-1); }
  .md :global(p) { margin: 0 0 6px; }
  .md :global(p:last-child) { margin-bottom: 0; }
  .md :global(ul), .md :global(ol) { margin: 4px 0 6px; padding-inline-start: 20px; }
  .md :global(li) { margin: 2px 0; }
  .md :global(table) {
    border-collapse: collapse;
    margin: 6px 0;
    font-size: var(--calcite-font-size--1, 13px);
  }
  .md :global(th), .md :global(td) {
    padding: 4px 12px;
    border-bottom: 1px solid var(--calcite-color-border-3);
    text-align: start;
  }
  .md :global(th) { color: var(--calcite-color-text-2); font-weight: var(--calcite-font-weight-medium, 500); }
  .md :global(td:last-child), .md :global(th:last-child) { text-align: end; }
  .md :global(code) {
    font-family: var(--calcite-code-family, monospace);
    font-size: 0.9em;
    padding: 1px 4px;
    border-radius: 3px;
    background: var(--calcite-color-foreground-3);
  }

  .metricas {
    font-size: var(--calcite-font-size--2, 11px);
    color: var(--calcite-color-text-3);
  }

  .paso {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 6px 0;
    border-bottom: 1px solid var(--calcite-color-border-3);
    font-size: var(--calcite-font-size--2, 11px);
    word-break: break-word;
  }

  .paso:last-child { border-bottom: none; }
  .paso strong { color: var(--calcite-color-brand); }
  .entrada { color: var(--calcite-color-text-2); }
  .salida { color: var(--calcite-color-text-3); }

  .pensando {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--calcite-color-text-3);
    font-size: var(--calcite-font-size--1, 13px);
  }

  .compositor {
    display: flex;
    flex-direction: column;
    gap: 6px;
    width: 100%;
  }

  .compositor calcite-text-area { width: 100%; }

  .acciones {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .ayuda {
    color: var(--calcite-color-text-3);
    font-size: var(--calcite-font-size--2, 12px);
  }
</style>