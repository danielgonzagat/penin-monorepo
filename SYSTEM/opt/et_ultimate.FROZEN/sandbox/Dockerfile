FROM python:3.10-slim
RUN useradd -m runner
WORKDIR /home/runner
# pacotes úteis mas seguros
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
USER runner
