#!/usr/bin/env python3
import logging, time, signal, sys, os
os.makedirs('/var/log/et_ultimate', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler('/var/log/et_ultimate/ultimate.log'),
              logging.StreamHandler()]
)
log = logging.getLogger("ET-ULTIMATE")
log.info("🚀 ET★★★★ Ultimate Core iniciado")
def handle(sig, frame):
    log.info("🛑 Encerrando com segurança..."); sys.exit(0)
signal.signal(signal.SIGINT, handle); signal.signal(signal.SIGTERM, handle)
while True:
    log.info("✅ heartbeat: ok")
    time.sleep(60)
