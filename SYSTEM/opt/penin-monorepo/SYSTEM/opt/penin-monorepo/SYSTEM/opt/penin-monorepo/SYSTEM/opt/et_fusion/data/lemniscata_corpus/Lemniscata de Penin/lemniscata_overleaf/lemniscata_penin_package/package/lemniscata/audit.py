# -*- coding: utf-8 -*-
import csv, os, time
from typing import Optional

class AuditLogger:
    def __init__(self, path: str):
        self.path = path
        self._ensure_header()

    def _ensure_header(self):
        exists = os.path.exists(self.path)
        if not exists:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp","E","N","I","iN","P","mode","reason"])

    def log(self, E: float, N: float, I: float, iN: float, P: float, mode: str, reason: Optional[str] = None):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([int(time.time()), f"{E:.6f}", f"{N:.6f}", f"{I:.6f}", f"{iN:.6f}", f"{P:.6f}", mode, reason or ""])
