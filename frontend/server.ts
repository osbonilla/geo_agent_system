/**
 * server.ts
 * ─────────
 * Servidor de producción del frontend, en Bun.
 *
 * Hace dos cosas:
 *   1. sirve el build estático de dist/
 *   2. proxea /api hacia el backend FastAPI
 *
 * El proxy evita que la URL del backend quede quemada en el bundle
 * y saca el CORS del medio: para el navegador todo es el mismo origen.
 *
 *   bun run server.ts
 */
const DIST = `${import.meta.dir}/dist`;
const PORT = Number(process.env.PORT ?? 5173);
const API_URL = (process.env.API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

const server = Bun.serve({
  port: PORT,
  hostname: '0.0.0.0',
  idleTimeout: 255, // las respuestas de los agentes pueden tardar

  async fetch(req) {
    const url = new URL(req.url);

    // ── Proxy hacia el backend ──────────────────────────
    if (url.pathname.startsWith('/api') || url.pathname === '/health') {
      const target = `${API_URL}${url.pathname}${url.search}`;
      try {
        const upstream = await fetch(target, {
          method: req.method,
          headers: req.headers,
          body: req.method === 'GET' || req.method === 'HEAD' ? undefined : await req.arrayBuffer(),
        });
        return new Response(upstream.body, {
          status: upstream.status,
          headers: upstream.headers,
        });
      } catch (err) {
        return Response.json(
          { detail: `El backend no responde en ${API_URL}. ${err}` },
          { status: 502 },
        );
      }
    }

    // ── Estáticos ───────────────────────────────────────
    const safePath = url.pathname.replace(/\.\./g, '');
    const file = Bun.file(`${DIST}${safePath}`);
    if (await file.exists()) {
      return new Response(file, {
        headers: safePath.startsWith('/assets/')
          ? { 'cache-control': 'public, max-age=31536000, immutable' }
          : {},
      });
    }

    // ── SPA fallback ────────────────────────────────────
    return new Response(Bun.file(`${DIST}/index.html`), {
      headers: { 'content-type': 'text/html; charset=utf-8' },
    });
  },
});

console.log(`GeoAgent web en http://localhost:${server.port}  ·  API → ${API_URL}`);
