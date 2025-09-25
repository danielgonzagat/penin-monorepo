FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN apt-get update -y && apt-get install -y --no-install-recommends     git curl build-essential procps nano openssh-client iproute2 iputils-ping     redis-server netcat-traditional && rm -rf /var/lib/apt/lists/*
RUN python -m pip install -U pip wheel setuptools &&     python -m pip install uvicorn fastapi pydantic==2.* rich loguru typer       numpy scipy pandas scikit-learn networkx sympy       "ray[default]==2.33.0" prometheus-client       torch --index-url https://download.pytorch.org/whl/cpu &&     python -m pip install transformers accelerate datasets sentencepiece peft evaluate
WORKDIR /app
COPY et_core.py /app/et_core.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
EXPOSE 7009 8266
CMD ["/app/entrypoint.sh"]
