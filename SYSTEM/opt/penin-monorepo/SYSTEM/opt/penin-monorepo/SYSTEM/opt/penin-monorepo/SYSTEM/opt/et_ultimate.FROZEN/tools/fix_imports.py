#!/usr/bin/env python3
import re, shutil, sys, time
from pathlib import Path

ROOT = Path("/opt/et_ultimate/agents/brain")
BACKUP_DIR = ROOT / "backups" / f"imports_fix_{time.strftime('%Y%m%d_%H%M%S')}"

PATTERNS = [
    # from et_mod import X -> from agents.brain.et_mod import X
    (re.compile(r"^(\s*)from\s+(et_[A-Za-z0-9_]+)\s+import\s+", re.M),
     r"\1from agents.brain.\2 import "),
    # from .et_mod import X -> from agents.brain.et_mod import X
    (re.compile(r"^(\s*)from\s+\.\s*(et_[A-Za-z0-9_]+)\s+import\s+", re.M),
     r"\1from agents.brain.\2 import "),
    # import et_mod as alias -> import agents.brain.et_mod as alias
    (re.compile(r"^(\s*)import\s+(et_[A-Za-z0-9_]+)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", re.M),
     r"\1import agents.brain.\2 as \3"),
    # import et_mod -> from agents.brain import et_mod
    (re.compile(r"^(\s*)import\s+(et_[A-Za-z0-9_]+)\s*$", re.M),
     r"\1from agents.brain import \2"),
]

def should_skip(p: Path) -> bool:
    s = str(p)
    return (
        not s.endswith(".py")
        or "/backups/" in s
        or s.endswith("__init__.py")  # deixa __init__ quieto
    )

def fix_text(txt: str) -> str:
    # não mexe se já é agents.brain
    out = txt
    for pat, repl in PATTERNS:
        out = pat.sub(repl, out)
    return out

def main():
    files = sorted(ROOT.rglob("*.py"))
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    changed = 0
    for f in files:
        if should_skip(f):
            continue
        src = f.read_text(encoding="utf-8")
        fixed = fix_text(src)
        if fixed != src:
            rel = f.relative_to(ROOT)
            (BACKUP_DIR / rel.parent).mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, BACKUP_DIR / rel)  # backup
            f.write_text(fixed, encoding="utf-8")
            print(f"[fix] {rel}")
            changed += 1
    print(f"\n✅ imports verificados. Arquivos alterados: {changed}")
    if changed:
        print(f"📦 backup em: {BACKUP_DIR}")

if __name__ == "__main__":
    sys.exit(main())
