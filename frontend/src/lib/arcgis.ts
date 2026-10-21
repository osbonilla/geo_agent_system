/**
 * lib/arcgis.ts
 * ─────────────
 * Carga el SDK de ArcGIS Maps for JavaScript desde el CDN de Esri
 * y expone un require() con promesas.
 *
 * Se carga por CDN a propósito: el paquete npm @arcgis/core pesa cientos
 * de megas y no aporta nada acá, porque el visor usa cuatro módulos.
 */
const VERSION = import.meta.env.VITE_ARCGIS_JS_VERSION ?? '4.34';
const BASE = `https://js.arcgis.com/${VERSION}`;

let loading: Promise<void> | null = null;

declare global {
  interface Window {
    require?: (mods: string[], cb: (...m: any[]) => void, err?: (e: any) => void) => void;
  }
}

function loadSdk(): Promise<void> {
  if (loading) return loading;

  loading = new Promise((resolve, reject) => {
    if (window.require) return resolve();

    const modo = document.body.classList.contains('calcite-mode-light') ? 'light' : 'dark';
    const css = document.createElement('link');
    css.id = 'esri-theme';
    css.rel = 'stylesheet';
    css.href = `${BASE}/esri/themes/${modo}/main.css`;
    document.head.appendChild(css);

    const script = document.createElement('script');
    script.src = `${BASE}/init.js`;
    script.onload = () => resolve();
    script.onerror = () =>
      reject(new Error(`No se pudo cargar el SDK de ArcGIS desde ${BASE}. Revisá la conexión.`));
    document.head.appendChild(script);
  });

  return loading;
}

export async function esri<T extends any[]>(modules: string[]): Promise<T> {
  await loadSdk();
  return new Promise((resolve, reject) => {
    window.require!(modules, (...mods: any[]) => resolve(mods as T), reject);
  });
}

/** Convierte un FeatureCollection en una capa GeoJSON con estilo del sistema. */
export async function geojsonLayer(fc: GeoJSON.FeatureCollection, title = 'Resultado') {
  const [GeoJSONLayer] = await esri<[any]>(['esri/layers/GeoJSONLayer']);
  const blob = new Blob([JSON.stringify(fc)], { type: 'application/json' });

  const geometry = fc.features[0]?.geometry?.type ?? 'Point';
  const renderer = {
    type: 'simple',
    symbol:
      geometry.includes('Polygon')
        ? { type: 'simple-fill', color: [0, 122, 194, 0.22],
            outline: { color: [0, 122, 194, 1], width: 1.4 } }
        : geometry.includes('Line')
        ? { type: 'simple-line', color: [0, 122, 194, 1], width: 2.2 }
        : { type: 'simple-marker', size: 8, color: [0, 122, 194, 0.9],
            outline: { color: [255, 255, 255, 1], width: 1.2 } },
  };

  return new GeoJSONLayer({
    url: URL.createObjectURL(blob),
    title,
    renderer,
    popupTemplate: { title, content: '{*}' },
  });
}
