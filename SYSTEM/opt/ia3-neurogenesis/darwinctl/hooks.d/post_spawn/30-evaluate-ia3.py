#!/usr/bin/env python3
import os, json, sys, time
sys.path.append("/opt/ia3-neurogenesis")

try:
    from neurogenesis.ia3_neurogenesis import IA3NeuronAgent
    
    neuron_id = os.getenv("NEURON_ID", "unknown")
    print(f"🔬 Avaliando neurônio {neuron_id}...")
    
    agent = IA3NeuronAgent(
        in_dim=16, 
        hidden_dim=8, 
        out_dim=16,
        metrics_port=int(os.getenv("IA3_METRICS_PORT", "9091"))
    )
    
    # Executar rodada IA³
    report = agent.round(seed=int(time.time()) % 10000)
    
    # Log resultado
    record = {
        "event": "neuron_evaluation", 
        "neuron_id": neuron_id, 
        "report": {
            "adaptive_gain": report.adaptive_gain,
            "alpha_shift": report.alpha_shift,
            "improved": report.improved,
            "structure_grew": report.structure_grew,
            "passes": report.passes,
            "consciousness_score": report.consciousness_score
        }
    }
    
    print(json.dumps(record, indent=2))
    
    # Critério DARWIN: neurônio deve provar IA³-like
    if not report.passes:
        print(f"❌ Neurônio {neuron_id} FALHOU na prova IA³")
        sys.exit(3)  # Código especial para falha IA³
    else:
        print(f"✅ Neurônio {neuron_id} APROVADO na prova IA³")

except Exception as e:
    print(f"💥 Erro na avaliação: {e}")
    sys.exit(1)
