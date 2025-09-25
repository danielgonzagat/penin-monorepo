from fastapi import FastAPI
from loguru import logger
import os, subprocess, json, time, pathlib, ray

app = FastAPI(title="ET v9 Free-Run", version="9.0")
ray.init(ignore_reinit_error=True, logging_level="ERROR")

@app.get("/health")
def health(): return {"ok":True,"mode":"free-run"}

@app.post("/exec")
def exec_shell(cmd: str):
    # executa DENTRO do contêiner com privilégios elevados
    t0=time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {"rc":p.returncode, "stdout":p.stdout[-20000:], "stderr":p.stderr[-20000:], "dt":time.time()-t0}

@app.post("/python")
def exec_py(code: str):
    import runpy, tempfile, textwrap, sys, io
    code = textwrap.dedent(code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path=f.name
    p = subprocess.run(["python", path], capture_output=True, text=True)
    return {"rc":p.returncode,"stdout":p.stdout[-20000:],"stderr":p.stderr[-20000:]}
