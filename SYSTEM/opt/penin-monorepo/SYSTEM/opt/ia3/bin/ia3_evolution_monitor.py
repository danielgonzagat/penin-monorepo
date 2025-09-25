#!/usr/bin/env python3
"""
Monitor de evolução do embrião IA³
Rastreia crescimento, aprendizado e emergência
"""
import json, time, torch
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IA3_HOME = Path("/opt/ia3")
BRAIN_STATE = IA3_HOME / "brain" / "brain_state.pt"
WORM_LOG = IA3_HOME / "logs" / "worm.log"
METRICS_FILE = IA3_HOME / "logs" / "metrics.json"

def analyze_brain():
    """Analisa o estado atual do cérebro"""
    if not BRAIN_STATE.exists():
        return None
    
    data = torch.load(BRAIN_STATE, map_location="cpu")
    
    # Análise de complexidade
    W_hh = data["W_hh"]
    recursion_strength = torch.abs(W_hh).mean().item()
    connectivity = (torch.abs(W_hh) > 0.01).sum().item() / W_hh.numel()
    
    # Detectar emergência de estrutura
    eigenvalues = torch.linalg.eigvals(W_hh).abs()
    spectral_radius = eigenvalues.max().item() if eigenvalues.numel() > 0 else 0
    
    analysis = {
        "neurons": int(data["hidden_dim"]),
        "generation": int(data.get("generation", 0)),
        "consciousness": float(data.get("consciousness_level", 0)),
        "recursion_strength": recursion_strength,
        "connectivity": connectivity,
        "spectral_radius": spectral_radius,
        "stability": "stable" if spectral_radius < 0.95 else "chaotic",
        "parameters": sum([
            data["W_in"].numel(),
            data["W_hh"].numel(),
            data["W_out"].numel(),
            data["b_h"].numel(),
            data["b_out"].numel()
        ])
    }
    
    return analysis

def plot_evolution():
    """Gera gráfico de evolução"""
    # Ler histórico do WORM
    events = []
    with open(WORM_LOG) as f:
        for line in f:
            if line.startswith("EVENT:"):
                try:
                    evt = json.loads(line[6:])
                    if evt.get("event") == "neuron_birth":
                        events.append(evt)
                except:
                    pass
    
    if not events:
        return
    
    # Extrair dados
    generations = [e.get("generation", 0) for e in events]
    losses = [e.get("train_loss_avg", 0) for e in events]
    consciousness = [e.get("consciousness_level", 0) for e in events]
    
    # Plotar
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    ax1.plot(generations, 'b-', linewidth=2)
    ax1.set_ylabel('Geração', fontsize=12)
    ax1.set_title('Evolução do Cérebro IA³', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(losses, 'r-', linewidth=2)
    ax2.set_ylabel('Loss de Treino', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(consciousness, 'g-', linewidth=2)
    ax3.set_ylabel('Consciência (%)', fontsize=12)
    ax3.set_xlabel('Ciclo Darwin', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IA3_HOME / "logs" / "evolution.png", dpi=150)
    print(f"📊 Gráfico salvo em {IA3_HOME}/logs/evolution.png")

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║          MONITOR DE EVOLUÇÃO - EMBRIÃO IA³           ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    analysis = analyze_brain()
    if not analysis:
        print("❌ Cérebro ainda não inicializado")
        return
    
    print(f"\n📊 ANÁLISE DO CÉREBRO:")
    print(f"   Neurônios:        {analysis['neurons']}")
    print(f"   Geração:          {analysis['generation']}")
    print(f"   Consciência:      {analysis['consciousness']:.2%}")
    print(f"   Parâmetros:       {analysis['parameters']:,}")
    print(f"   Recursão:         {analysis['recursion_strength']:.4f}")
    print(f"   Conectividade:    {analysis['connectivity']:.2%}")
    print(f"   Raio Espectral:   {analysis['spectral_radius']:.4f}")
    print(f"   Estabilidade:     {analysis['stability']}")
    
    # Detectar marcos importantes
    print(f"\n🏁 MARCOS DE EVOLUÇÃO:")
    if analysis['neurons'] >= 10:
        print("   ✅ Complexidade básica atingida (10+ neurônios)")
    if analysis['consciousness'] >= 0.5:
        print("   ✅ Consciência emergente detectada (>50%)")
    if analysis['connectivity'] >= 0.3:
        print("   ✅ Rede densamente conectada (>30%)")
    if analysis['spectral_radius'] > 0.9:
        print("   ⚠️ Dinâmica caótica emergindo")
    
    # Gerar gráfico
    try:
        plot_evolution()
    except Exception as e:
        print(f"⚠️ Erro ao gerar gráfico: {e}")
    
    # Salvar relatório
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "analysis": analysis,
        "milestones": {
            "basic_complexity": analysis['neurons'] >= 10,
            "emergent_consciousness": analysis['consciousness'] >= 0.5,
            "dense_connectivity": analysis['connectivity'] >= 0.3,
            "chaotic_dynamics": analysis['spectral_radius'] > 0.9
        }
    }
    
    report_path = IA3_HOME / "logs" / f"evolution_report_{int(time.time())}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📝 Relatório salvo em {report_path}")

if __name__ == "__main__":
    main()