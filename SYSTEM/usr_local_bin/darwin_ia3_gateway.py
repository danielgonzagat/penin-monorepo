#!/usr/bin/env python3
"""
Darwin ↔ IA³ Gateway
Conecta decisões Darwin com o embrião neural IA³
"""
import os, sys, json, subprocess, hashlib, time
from pathlib import Path
from datetime import datetime

# Paths
NEURONGEN = "/opt/ia3/bin/neurongen.py"
WORM_IA3 = "/opt/ia3/logs/worm.log"
WORM_DARWIN = "/root/darwin_worm.log"
PROMOTION_LOG = "/root/promotion_log.json"

def append_worm(path, event):
    """Append to WORM with hash chain"""
    event["timestamp"] = datetime.utcnow().isoformat() + "Z"
    
    # Get previous hash
    prev_hash = "GENESIS"
    if os.path.exists(path):
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if line.startswith("HASH:"):
                    prev_hash = line.split("HASH:", 1)[1].strip()
                    break
    
    event["previous_hash"] = prev_hash
    payload = json.dumps(event, ensure_ascii=False)
    curr_hash = hashlib.sha256((prev_hash + payload).encode()).hexdigest()
    
    with open(path, 'a') as f:
        f.write(f"EVENT:{payload}\n")
        f.write(f"HASH:{curr_hash}\n")
    
    return curr_hash

def read_last_promotion():
    """Read last promotion decision from promotion_log"""
    if not os.path.exists(PROMOTION_LOG):
        return None
    
    with open(PROMOTION_LOG, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            try:
                data = json.loads(line)
                return data
            except:
                continue
    return None

def trigger_neurongen(event="live", mode="numeric"):
    """Call neurongen with Darwin event"""
    cmd = [
        "python3", NEURONGEN,
        "--event", event,
        "--mode", mode,
        "--metrics-port", "9101",
        "--worm", WORM_IA3,
        "--seed", str(int(time.time()) % 10000)
    ]
    
    print(f"🔧 Executando: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        # Parse JSON output
        try:
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if line.startswith('{'):
                    return json.loads(line)
        except:
            pass
    else:
        print(f"❌ Erro: {result.stderr}")
    
    return {"ok": False, "error": result.stderr}

def main():
    print("\n" + "="*60)
    print("🔀 DARWIN ↔ IA³ GATEWAY")
    print("="*60)
    
    # Read last Darwin decision
    promo = read_last_promotion()
    
    if not promo:
        print("⚠️ Sem decisões no promotion_log, usando default")
        decision = "live"
        metrics = {}
    else:
        print(f"📊 Última decisão Darwin:")
        print(f"   Timestamp: {promo.get('timestamp', 'unknown')}")
        
        # Extract decision
        if promo.get("promote", False) or promo.get("survive", False):
            decision = "live"
            print(f"   ✅ Decisão: VIVE (promote={promo.get('promote')}, survive={promo.get('survive')})")
        else:
            decision = "death"
            print(f"   ☠️ Decisão: MORTE")
        
        metrics = promo.get("metrics", {})
        if metrics:
            print(f"   ΔL∞: {metrics.get('delta_linf', 'N/A')}")
            print(f"   I: {metrics.get('I', 'N/A')}")
            print(f"   P: {metrics.get('P', 'N/A')}")
            print(f"   CAOS: {metrics.get('caos_ratio', 'N/A')}")
    
    # Trigger IA³ neurongen
    print(f"\n🧬 Aplicando ao embrião IA³...")
    result = trigger_neurongen(event=decision, mode="numeric")
    
    # Log to Darwin WORM
    gateway_event = {
        "event": "darwin_ia3_gateway",
        "darwin_decision": decision,
        "darwin_metrics": metrics,
        "ia3_result": result,
        "trigger": "manual"
    }
    
    hash_darwin = append_worm(WORM_DARWIN, gateway_event)
    hash_ia3 = append_worm(WORM_IA3, gateway_event)
    
    print(f"\n📝 WORM Logs:")
    print(f"   Darwin: {hash_darwin[:16]}...")
    print(f"   IA³: {hash_ia3[:16]}...")
    
    print("\n" + "="*60)
    print("✅ Gateway executado com sucesso")
    print("="*60 + "\n")
    
    return 0 if result.get("ok") else 1

if __name__ == "__main__":
    sys.exit(main())