import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// En desarrollo Vite proxea /api al backend, igual que hace server.ts
// en producción. Así el frontend nunca conoce la URL del backend.
const API_URL = process.env.API_URL ?? 'http://localhost:8000';

export default defineConfig({
  plugins: [svelte()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': { target: API_URL, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    target: 'esnext',
    sourcemap: false,
  },
});
