FROM python:3.11-slim
RUN apt-get update -y && apt-get install -y --no-install-recommends       git curl build-essential procps nano openssh-client &&     rm -rf /var/lib/apt/lists/*
WORKDIR /app
# (Você pode clonar seu repositório da ETΩ/ET★★★★ aqui no futuro)
RUN python -m pip install --no-cache-dir uvicorn fastapi pydantic==2.*     && python -m pip install --no-cache-dir numpy scipy pandas scikit-learn     && python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Diretórios úteis
RUN mkdir -p /workspace /changes_out
ENV ET_MODE=propose-only
ENV PYTHONUNBUFFERED=1
# Entrypoint simples: servidor + agente
COPY et_core.py /app/et_core.py
CMD ["python","/app/et_core.py"]
