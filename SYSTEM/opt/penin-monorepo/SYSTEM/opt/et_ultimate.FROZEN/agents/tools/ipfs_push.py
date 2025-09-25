import os
from pathlib import Path
from subprocess import check_output

HISTORY = Path("/opt/et_ultimate/history")
for f in HISTORY.glob("*.py"):
    cid = check_output(["ipfs", "add", "-Q", str(f)]).decode().strip()
    print(f"🧬 {f.name} → IPFS CID: {cid}")
