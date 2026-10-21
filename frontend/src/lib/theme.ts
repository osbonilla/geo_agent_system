/**
 * lib/theme.ts
 * ────────────
 * Modo claro / oscuro coordinado entre Calcite (clase en <body>) y el
 * SDK de mapas (hoja de estilos esri/themes/<modo>/main.css).
 */
export type Modo = 'dark' | 'light';

const KEY = 'geoagent.modo';

export function modoInicial(): Modo {
  try {
    const guardado = localStorage.getItem(KEY);
    if (guardado === 'dark' || guardado === 'light') return guardado;
  } catch { /* almacenamiento no disponible */ }
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export function aplicarModo(modo: Modo) {
  document.body.classList.toggle('calcite-mode-dark', modo === 'dark');
  document.body.classList.toggle('calcite-mode-light', modo === 'light');
  const link = document.getElementById('esri-theme') as HTMLLinkElement | null;
  if (link) link.href = link.href.replace(/themes\/(dark|light)\//, `themes/${modo}/`);
  try {
    localStorage.setItem(KEY, modo);
  } catch { /* almacenamiento no disponible */ }
}
