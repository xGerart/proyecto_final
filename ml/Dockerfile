# Imagen de trabajo de ML: la usan las tareas de Kestra (lotes, monitoreo, reentrenamiento)
# y el tablero. El código (src/, monitoreo/, tablero/) se monta desde el repositorio, así
# que un cambio de código no exige reconstruir la imagen.
FROM python:3.12.12-slim-bookworm

# git: cada corrida de MLflow registra el commit del código que la generó.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory /app

COPY ml/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
