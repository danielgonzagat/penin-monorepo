#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETΩ Watchdog — compat layer
Expõe as funções esperadas pelo cérebro:
- iniciar_watchdog()
- marcar_progresso()
- relatar_falha()

Implementa monitoramento simples de estagnação e heartbeats.
Sem dependências externas.
"""

from __future__ import annotations
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

LOG_DIR = Path(os.environ.get("ET_LOG_DIR", "/opt/et_ultimate/logs"))
STATE_DIR = Path(os.environ.get("ET_STATE_DIR", "/opt/et_ultimate/state"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

HEARTBEAT = STATE_DIR / "watchdog_heartbeat.json"
EVENTS_LOG = LOG_DIR / "watchdog.log"

DEFAULT_STALL_SECONDS = int(os.environ.get("ET_WATCH_STALL_SEC", "900"))  # 15 min
DEFAULT_TICK_SECONDS = int(os.environ.get("ET_WATCH_TICK_SEC", "5"))     # 5 s

_lock = threading.Lock()
_watchdog_thread: Optional[threading.Thread] = None
_watchdog_stop = threading.Event()

@dataclass
class WatchState:
    last_task: str = "boot"
    last_detail: str = ""
    last_ts: float = field(default_factory=lambda: time.time())
    counters: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "last_task": self.last_task,
            "last_detail": self.last_detail,
            "last_ts": self.last_ts,
            "counters": self.counters,
        }

_state = WatchState()

def _log_event(level: str, msg: str, extra: Optional[Dict]=None) -> None:
    EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "level": level,
        "msg": msg,
    }
    if extra:
        payload.update(extra)
    with EVENTS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _write_heartbeat() -> None:
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    tmp = HEARTBEAT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_state.to_dict(), f, ensure_ascii=False)
    tmp.replace(HEARTBEAT)

def marcar_progresso(task: str, detail: str = "") -> None:
    """API esperada pelo cérebro: registra avanço de etapa."""
    with _lock:
        _state.last_task = task
        _state.last_detail = detail
        _state.last_ts = time.time()
        _state.counters[task] = _state.counters.get(task, 0) + 1
        _write_heartbeat()
    _log_event("info", f"progress:{task}", {"detail": detail})

def relatar_falha(contexto: str, erro: str, severidade: str = "error") -> None:
    """API esperada: registra falha diagnóstica."""
    with _lock:
        _state.last_detail = f"FAIL[{severidade}]: {contexto}"
        _write_heartbeat()
    _log_event(severidade, f"failure:{contexto}", {"error": erro})

def _watch_loop(stall_seconds: int, tick_seconds: int) -> None:
    _log_event("info", "watchdog-start", {
        "stall_seconds": stall_seconds,
        "tick_seconds": tick_seconds
    })
    while not _watchdog_stop.is_set():
        time.sleep(tick_seconds)
        try:
            with _lock:
                since = time.time() - _state.last_ts
                snapshot = _state.to_dict()
            if since >= stall_seconds:
                # assinala estagnação
                _log_event(
                    "warn",
                    "stagnation-detected",
                    {"idle_seconds": since, "snapshot": snapshot}
                )
                # toca o sino: escreve um arquivo de sinalização que o cérebro pode ler
                (STATE_DIR / "stagnation.flag").write_text(
                    json.dumps({"ts": time.time(), "idle": since, "snapshot": snapshot}),
                    encoding="utf-8"
                )
                # após sinalizar, reinicia o marcador para evitar flood
                with _lock:
                    _state.last_ts = time.time()
                    _write_heartbeat()
        except Exception as e:
            _log_event("error", "watchdog-loop-exception", {"error": repr(e)})

    _log_event("info", "watchdog-stop", {})

def iniciar_watchdog(
    stall_seconds: int = DEFAULT_STALL_SECONDS,
    tick_seconds: int = DEFAULT_TICK_SECONDS
) -> None:
    """API esperada: sobe o watchdog em background, idempotente."""
    global _watchdog_thread
    if _watchdog_thread and _watchdog_thread.is_alive():
        _log_event("info", "watchdog-already-running", {})
        return
    _watchdog_stop.clear()
    _watchdog_thread = threading.Thread(
        target=_watch_loop, args=(stall_seconds, tick_seconds), daemon=True
    )
    _watchdog_thread.start()
    marcar_progresso("watchdog.boot", f"stall={stall_seconds}s tick={tick_seconds}s")

# aliases opcionais (se algum trecho antigo importar esses nomes):
start_watchdog = iniciar_watchdog
mark_progress = marcar_progresso
report_failure = relatar_falha

if __name__ == "__main__":
    # quick self-test
    iniciar_watchdog(stall_seconds=10, tick_seconds=1)
    for i in range(3):
        marcar_progresso("selftest.step", f"i={i}")
        time.sleep(0.5)
    time.sleep(12)  # força detecção de estagnação
    relatar_falha("selftest", "fim do autoteste", "info")
def loop_guard(*args, **kwargs):
    """
    Wrapper de compatibilidade para manter módulos antigos funcionando.
    O loop_guard garante monitoramento contínuo e prevenção de loops ou estagnação.
    """
    return iniciar_watchdog(*args, **kwargs)

def enforce_novelty(*args, **kwargs):
    """
    Wrapper para manter compatibilidade com enforce_novelty esperado por módulos antigos.
    Pode ser adaptado para acionar a lógica atual de verificação de novidade.
    """
    # Se houver função equivalente no código atual, chame-a aqui
    if 'verificar_novidade' in globals():
        return verificar_novidade(*args, **kwargs)
    return True
