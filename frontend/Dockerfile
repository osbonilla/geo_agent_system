# ══════════════════════════════════════════════════════════
#  Frontend — Svelte 5: dependencias con Bun, build con Vite
#  sobre Node, y servido en producción con Bun (server.ts)
# ══════════════════════════════════════════════════════════

# ── Build ─────────────────────────────────────────────────
FROM oven/bun:1 AS build

# La imagen oven/bun no trae Node. Sin Node, `bun run build` ejecuta Vite
# con el runtime de Bun, que a veces no resuelve `svelte` desde
# @sveltejs/vite-plugin-svelte:
#   "Cannot find package 'svelte' imported from .../vite-plugin-svelte/..."
# Con Node presente, `bun run` respeta el shebang de Vite y compila con Node,
# que es el runtime que Vite soporta oficialmente.
COPY --from=node:22-slim /usr/local/bin/node /usr/local/bin/node

WORKDIR /app
ENV NODE_ENV=development

COPY package.json ./
COPY bun.lock* ./

# --linker hoisted: node_modules plano, sin enlaces simbólicos (el modo
# "isolated" de Bun 1.3+ es otra fuente de errores de resolución).
# Si el lockfile lo escribió otra versión de Bun y falla, se instala sin él.
RUN set -eux; \
    bun install --linker hoisted || { rm -f bun.lock; bun install --linker hoisted; }; \
    node -e "import('svelte/compiler').then(() => console.log('svelte OK'))"

COPY . .

ARG ARCGIS_JS_VERSION=4.34
ENV VITE_ARCGIS_JS_VERSION=$ARCGIS_JS_VERSION

RUN bun run build && test -f dist/index.html

# ── Runtime ───────────────────────────────────────────────
# Solo viajan dist/ y server.ts: la imagen final no lleva node_modules.
FROM oven/bun:1-slim AS runtime

WORKDIR /app

COPY --from=build /app/dist ./dist
COPY --from=build /app/server.ts ./server.ts

ENV NODE_ENV=production \
    PORT=5173 \
    API_URL=http://api:8000

EXPOSE 5173

USER bun
CMD ["bun", "run", "server.ts"]
