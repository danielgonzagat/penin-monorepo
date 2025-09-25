# /opt/et_ultimate/agents/brain/et_guardian.py
# Hardening em camadas: bloqueia rede, subprocessos e escrita fora de paths permitidos.
# Ativa automaticamente ao importar este módulo no processo do cérebro.

import os, sys, types, builtins, socket, subprocess, pathlib, errno

ET_GUARD_ON = True
os.environ.setdefault("ET_LOCKED", "1")
os.environ.setdefault("ET_OFFLINE", "1")
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("HTTP_PROXY", "")
os.environ.setdefault("HTTPS_PROXY", "")

# Áreas explicitamente permitidas
ROOT = pathlib.Path("/opt/et_ultimate").resolve()
ALLOW_READONLY = [
    ROOT / "agents",
    ROOT / "venv",
    pathlib.Path("/usr/lib/python3.10"),
    pathlib.Path("/usr/local/lib/python3.10"),
]
ALLOW_WRITE = [
    ROOT / "workspace",
    ROOT / "history",
    ROOT / "logs",
    pathlib.Path("/tmp"),
]

def _norm(p) -> pathlib.Path:
    return pathlib.Path(p).resolve() if isinstance(p, (str, os.PathLike)) else p

def _is_under(path: pathlib.Path, base: pathlib.Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except Exception:
        return False

def _can_write(path: pathlib.Path) -> bool:
    path = _norm(path)
    return any(_is_under(path, b) for b in ALLOW_WRITE)

def _can_read(path: pathlib.Path) -> bool:
    path = _norm(path)
    return any(_is_under(path, b) for b in (ALLOW_READONLY + ALLOW_WRITE))

# --- BLOQUEIO DE ARQUIVOS (escrita fora da caixa)
_orig_open = builtins.open
def _safe_open(file, mode="r", *args, **kwargs):
    fpath = _norm(file)
    if any(m in mode for m in ("w", "a", "x", "+")):
        if not _can_write(fpath):
            raise PermissionError(f"ET-Guardian: escrita bloqueada em {fpath}")
    else:
        if not _can_read(fpath):
            raise PermissionError(f"ET-Guardian: leitura bloqueada em {fpath}")
    return _orig_open(file, mode, *args, **kwargs)

# --- BLOQUEIO DE SUBPROCESSOS
def _deny_subprocess(*a, **k):
    raise PermissionError("ET-Guardian: subprocessos proibidos neste contexto")

# --- BLOQUEIO DE REDE (qualquer socket de rede)
class _NoNetSocket:
    def __init__(self, *a, **k):
        raise PermissionError("ET-Guardian: acesso de rede bloqueado")

# --- INJEÇÃO DE MÓDULOS BLOQUEADOS (requests, httpx, etc.)
def _blocked_module(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    def _blocked(*a, **k):
        raise PermissionError(f"ET-Guardian: módulo '{name}' bloqueado")
    m.__getattr__ = lambda _attr: _blocked
    return m

BLOCKLIST_IMPORTS = [
    "requests", "httpx", "urllib", "urllib3", "websocket", "websockets",
    "smtplib", "ftplib", "imaplib", "poplib",
    "paramiko",
    "boto3", "botocore",
    "openai", "anthropic", "mistralai", "google", "vertexai", "grpc",
    "paho", "slack_sdk", "discord", "telegram", "twilio",
    "psycopg2", "pymysql", "pymongo", "redis", "kafka", "pulsar"
]

def harden():
    if not ET_GUARD_ON:
        return

    # Open
    builtins.open = _safe_open

    # Subprocess
    subprocess.Popen = _deny_subprocess
    subprocess.run = _deny_subprocess
    subprocess.call = _deny_subprocess
    subprocess.check_call = _deny_subprocess
    subprocess.check_output = _deny_subprocess
    os.system = _deny_subprocess

    # Socket
    socket.socket = _NoNetSocket  # qualquer tentativa gera PermissionError
    # (AF_UNIX não será possível via socket.socket; se precisar de IPC local, faça via arquivos.)

    # Pré-bloqueia módulos de rede
    for name in BLOCKLIST_IMPORTS:
        if name in sys.modules:
            sys.modules[name] = _blocked_module(name)
        else:
            sys.modules[name] = _blocked_module(name)

    # Garante que o código do cérebro só escreva onde pode
    for p in (ALLOW_WRITE):
        os.makedirs(p, exist_ok=True)

# Ativa imediatamente no import
harden()
