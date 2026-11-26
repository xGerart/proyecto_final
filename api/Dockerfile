# Servicio de predicción. Contexto de construcción: la raíz del repositorio.
# El modelo NO va dentro de la imagen: se descarga del Model Registry al arrancar
# (MODELO_URI), así que promover o revertir una versión no exige reconstruirla.
FROM python:3.12.12-slim-bookworm
LABEL org.opencontainers.image.source=https://github.com/xGerart/proyecto_final

RUN useradd --create-home --uid 10001 api
COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY src/__init__.py src/pipeline.py src/datos.py src/
COPY api/ api/
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
USER api
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/salud')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
