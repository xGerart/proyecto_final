# Atajos del proyecto. Requiere .env (copiar de .env.example) y las llaves en secrets/.
include .env
export

KESTRA   := http://localhost:8080/api/v1/main
NS       := proyecto_final
CURL     := curl -sS --fail-with-body -u "$(KESTRA_USER):$(KESTRA_PASSWORD)"
RUNNER   := docker run --rm --env-file .env.runtime -v $(PWD)/dbt:/app/dbt -v $(PWD)/kestra/scripts:/app/scripts -v $(PWD)/secrets:/app/secrets:ro proyecto-final-runner:1.0
PY       := .venv/bin/python

.PHONY: secrets build up down deploy ingest backfill transform dbt-build status \
        api-local test simulacion exportar

## Genera .env.kestra (SECRET_* en base64 para Kestra) y .env.runtime (para correr a mano).
secrets:
	@test -f "$(SNOWFLAKE_PRIVATE_KEY_FILE)" || (echo "Falta $(SNOWFLAKE_PRIVATE_KEY_FILE)"; exit 1)
	@test -f "$(ERP_SSH_KEY_FILE)" || (echo "Falta $(ERP_SSH_KEY_FILE)"; exit 1)
	@ssh-keyscan -t ed25519 $(ERP_SSH_HOST) 2>/dev/null > secrets/known_hosts
	@python3 kestra/scripts/make_secrets.py
	@echo "OK: .env.kestra y .env.runtime generados (no se versionan)."

## Imágenes locales: runner (ingesta + dbt), Spark, ML (tareas de Kestra + tablero) y MLflow.
build:
	docker compose --profile build build runner spark-master mlflow tablero

## Levanta toda la infraestructura: Kestra, Spark, MLflow, API y tablero.
up:
	docker compose up -d --wait
	@echo "Kestra http://localhost:8080 · Spark http://localhost:8081 · MLflow http://localhost:5050"
	@echo "API http://localhost:8010/docs · Tablero http://localhost:8501"

down:
	docker compose down

## La API desde el código local (misma imagen que publica cd.yml, con tag API_TAG=local).
api-local:
	API_TAG=local docker compose build api
	API_TAG=local docker compose up -d --wait api

## Publica (crea o actualiza) los flujos de kestra/*.yml en Kestra.
deploy:
	@for f in kestra/*.yml; do \
	  echo "→ $$f"; \
	  $(CURL) -X POST "$(KESTRA)/flows/validate" -H "Content-Type: application/x-yaml" --data-binary "@$$f" \
	    | python3 -c "import json,sys; e=[x['constraints'] for x in json.load(sys.stdin) if x.get('constraints')]; print(*e, sep='\\n'); sys.exit(1 if e else 0)" || exit 1; \
	  $(CURL) -X POST "$(KESTRA)/flows/import" -F "fileUpload=@$$f" >/dev/null || exit 1; \
	done

## Corrida diaria manual de la ingesta (ventana de reproceso de 7 días).
ingest:
	@$(CURL) -X POST "$(KESTRA)/executions/$(NS)/ingest_erp" | python3 -m json.tool | grep '"id"' | head -1

## Backfill histórico completo en una sola ejecución (al terminar dispara la transformación).
backfill:
	@$(CURL) -X POST "$(KESTRA)/executions/$(NS)/ingest_erp" \
	  -F start_date=$(ANALYSIS_START_DATE) -F end_date=$(ANALYSIS_END_DATE) | python3 -m json.tool | grep '"id"' | head -1

transform:
	@$(CURL) -X POST "$(KESTRA)/executions/$(NS)/transform" | python3 -m json.tool | grep '"id"' | head -1

## Simulación de producción: lotes -> predicciones -> monitoreo -> reentrenamiento.
simulacion:
	@$(CURL) -X POST "$(KESTRA)/executions/$(NS)/produccion_lotes" | python3 -m json.tool | grep '"id"' | head -1

## dbt fuera de Kestra (desarrollo).
dbt-build:
	$(RUNNER) sh -c 'export SNOWFLAKE_PRIVATE_KEY="$$(cat /app/secrets/snowflake_rsa_key.p8)"; cd /app/dbt && dbt deps && dbt build'

status:
	@$(CURL) "$(KESTRA)/executions/search?namespace=$(NS)&size=15" | python3 kestra/scripts/print_executions.py

## Pruebas (las mismas que corre ci.yml).
test:
	$(PY) -m pytest -q

## Exporta a docs/ las tablas de Snowflake y las corridas de MLflow.
exportar:
	$(PY) -m src.exportar
