# ══════════════════════════════════════════════════════════
#  GeoAgent System — atajos
#  Ejecutá `make` para ver la lista.
# ══════════════════════════════════════════════════════════
.DEFAULT_GOAL := help
.PHONY: help up down logs build rebuild ps shell vectorize dev-api dev-web install clean test eval

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up:  ## Levanta todo el stack (postgres, chroma, api, frontend)
	docker compose up -d --build

down:  ## Apaga el stack
	docker compose down

logs:  ## Sigue los logs del backend
	docker compose logs -f api

ps:  ## Estado de los contenedores
	docker compose ps

shell:  ## Abre una shell dentro del contenedor de la API
	docker compose exec api bash

vectorize:  ## Vectoriza todos los WebMaps del usuario
	docker compose exec api python -c "from tools.vector_tools import vectorize_all_webmaps; print(vectorize_all_webmaps(deep=True))"

test:  ## Tests offline (sin ArcGIS, LLM ni base de datos)
	python -m pytest tests/test_offline.py tests/test_spatial.py -q

eval:  ## Evaluación SLM vs LLM. Uso: make eval MODELOS=ollama/qwen2.5:7b,nvidia_nim/openai/gpt-oss-20b
	docker compose exec api python eval/run_eval.py --modelos "$(MODELOS)" --repeticiones $(or $(REP),1)

install:  ## Instala dependencias en local (Python + Bun)
	pip install -r requirements.txt
	cd frontend && bun install

dev-api:  ## Backend en modo desarrollo con recarga
	uvicorn api.main:app --reload --port 8000

dev-web:  ## Frontend en modo desarrollo con Bun
	cd frontend && bun run dev

clean:  ## Borra caches, build del frontend e índice local
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf frontend/dist frontend/node_modules chroma_data
