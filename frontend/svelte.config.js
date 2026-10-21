import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),
  compilerOptions: {
    // Los componentes Calcite (calcite-button, calcite-action...) ya son
    // interactivos y accesibles: los avisos a11y de Svelte no aplican a ellos.
    warningFilter: (w) => !(w.code?.startsWith('a11y') && /<calcite-/.test(w.message)),
  },
};
