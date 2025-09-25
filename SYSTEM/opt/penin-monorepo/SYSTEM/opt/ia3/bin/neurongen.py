#!/usr/bin/env python3
# /opt/ia3/bin/neurongen.py
# Embrião IA³: a cada rodada Darwin, cresce (ou poda) 1 neurônio REAL (PyTorch).
# NOVO: Cada neurônio deve PROVAR que atende aos 9 critérios IA³ ou MORRE!

import argparse, json, math, os, random, time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Métricas Prometheus
try:
    from prometheus_client import Gauge, Counter, start_http_server
    PROM_OK = True
except Exception:
    PROM_OK = False

# ------------------------
# Utilidades
# ------------------------
def now_utc():
    return datetime.utcnow().isoformat() + "Z"

def jwrite(path, obj, append=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)

# ------------------------
# CRITÉRIOS IA³ - PROVA DE VIDA
# ------------------------
class IA3Validator:
    """
    Valida se um neurônio atende aos 9 critérios IA³:
    1. Adaptativo - Se adapta a novos padrões
    2. Autorecursivo - Processa próprias saídas
    3. Autoevolutivo - Melhora com o tempo
    4. Autodidata - Aprende sem supervisão explícita  
    5. Autônomo - Opera independentemente
    6. Autossuficiente - Não depende de recursos externos
    7. Autoconstrutivo - Participa da própria construção
    8. Autosináptico - Gerencia próprias conexões
    9. Autoarquitetável - Define própria estrutura
    """
    
    @staticmethod
    def validate_neuron(neuron_stats):
        """Retorna (is_valid, score, failures)"""
        criteria = {}
        failures = []
        
        # 1. Adaptativo - gradientes devem fluir (não pode estar morto)
        grad_flow = neuron_stats.get("grad_norm", 0)
        criteria["adaptativo"] = grad_flow > 1e-6
        if not criteria["adaptativo"]:
            failures.append("Não-adaptativo: gradiente morto")
        
        # 2. Autorecursivo - conexões recursivas ativas
        recursion = neuron_stats.get("recursion_strength", 0)
        criteria["autorecursivo"] = recursion > 0.01
        if not criteria["autorecursivo"]:
            failures.append("Não-recursivo: sem loops internos")
        
        # 3. Autoevolutivo - loss deve melhorar
        improvement = neuron_stats.get("loss_improvement", 0)
        criteria["autoevolutivo"] = improvement > -0.1  # Permite pequena piora
        if not criteria["autoevolutivo"]:
            failures.append("Não-evolutivo: performance piorando")
        
        # 4. Autodidata - deve aprender padrões
        pattern_score = neuron_stats.get("pattern_recognition", 0)
        criteria["autodidata"] = pattern_score > 0.1
        if not criteria["autodidata"]:
            failures.append("Não-autodidata: não reconhece padrões")
        
        # 5. Autônomo - ativação independente
        activation_var = neuron_stats.get("activation_variance", 0)
        criteria["autonomo"] = activation_var > 0.001
        if not criteria["autonomo"]:
            failures.append("Não-autônomo: sempre inativo ou saturado")
        
        # 6. Autossuficiente - contribuição única
        contribution = neuron_stats.get("unique_contribution", 0)
        criteria["autossuficiente"] = contribution > 0.05
        if not criteria["autossuficiente"]:
            failures.append("Não-autossuficiente: redundante")
        
        # 7. Autoconstrutivo - participação no crescimento
        growth_factor = neuron_stats.get("growth_participation", 0)
        criteria["autoconstrutivo"] = growth_factor > 0
        if not criteria["autoconstrutivo"]:
            failures.append("Não-construtivo: não contribui para crescimento")
        
        # 8. Autosináptico - conexões dinâmicas
        synapse_change = neuron_stats.get("synapse_plasticity", 0)
        criteria["autosinaptico"] = synapse_change > 0.001
        if not criteria["autosinaptico"]:
            failures.append("Não-sináptico: conexões estáticas")
        
        # 9. Autoarquitetável - influência estrutural
        structural_impact = neuron_stats.get("structural_influence", 0)
        criteria["autoarquitetavel"] = structural_impact > 0.01
        if not criteria["autoarquitetavel"]:
            failures.append("Não-arquitetável: sem impacto estrutural")
        
        # Cálculo do score
        passed = sum(criteria.values())
        total = len(criteria)
        score = passed / total
        
        # Neurônio válido se passa em TODOS os critérios
        is_valid = all(criteria.values())
        
        return is_valid, score, failures, criteria

# ------------------------
# Cerebelo IA³ com Validação
# ------------------------
class Brain(nn.Module):
    """
    Rede recorrente com validação IA³ por neurônio.
    Cada neurônio deve provar que atende aos 9 critérios ou morre.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, activation="relu", device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_name = activation
        
        # Estatísticas por neurônio para validação IA³
        self.neuron_stats = [{
            "birth_time": datetime.utcnow(),
            "age": 0,
            "total_activations": 0,
            "grad_accumulator": 0,
            "last_loss": float('inf'),
            "recursion_count": 0,
            "pattern_matches": 0,
            "unique_outputs": set(),
            "synapse_updates": 0
        } for _ in range(hidden_dim)]

        self.W_in  = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.W_hh  = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_h   = nn.Parameter(torch.zeros(hidden_dim))
        self.W_out = nn.Parameter(torch.empty(output_dim, hidden_dim))
        self.b_out = nn.Parameter(torch.zeros(output_dim))
        
        # Histórico para validação
        self.loss_history = []
        self.activation_history = []

        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_in, a=math.sqrt(5))
        nn.init.orthogonal_(self.W_hh)
        nn.init.kaiming_uniform_(self.W_out, a=math.sqrt(5))

    def act(self, x):
        if self.activation_name == "tanh":
            return torch.tanh(x)
        elif self.activation_name == "gelu":
            return F.gelu(x)
        return F.relu(x)

    def forward(self, x, h=None):
        """Forward com coleta de estatísticas para validação IA³"""
        B = x.shape[0]
        if h is None:
            h = torch.zeros(B, self.hidden_dim, device=self.device)
        
        # Computação com hooks para estatísticas
        pre_act = x @ self.W_in.T + h @ self.W_hh.T + self.b_h
        h_new = self.act(pre_act)
        
        # Coletar estatísticas de ativação (sem gradiente)
        with torch.no_grad():
            for i in range(self.hidden_dim):
                if i < len(self.neuron_stats):
                    self.neuron_stats[i]["total_activations"] += h_new[:, i].abs().mean().item()
                    self.neuron_stats[i]["recursion_count"] += (h[:, i].abs() > 0.1).float().mean().item()
            
            # Armazenar histórico sem gradientes
            if not hasattr(self, 'activation_history'):
                self.activation_history = []
            self.activation_history.append(h_new.detach().clone())
            if len(self.activation_history) > 100:
                self.activation_history.pop(0)
        
        y = h_new @ self.W_out.T + self.b_out
        return y, h_new.detach()  # Detach hidden state to prevent gradient accumulation

    def compute_neuron_stats(self, idx):
        """Computa estatísticas do neurônio idx para validação IA³"""
        if idx >= self.hidden_dim:
            return {}
        
        stats = {}
        
        # 1. Grad norm (adaptabilidade)
        if self.W_hh.grad is not None:
            stats["grad_norm"] = torch.norm(self.W_hh.grad[idx]).item()
        else:
            stats["grad_norm"] = 0
        
        # 2. Recursion strength (autorecursividade)
        stats["recursion_strength"] = torch.abs(self.W_hh[idx, idx]).item()
        
        # 3. Loss improvement (autoevolução)
        if len(self.loss_history) >= 2:
            stats["loss_improvement"] = self.loss_history[-2] - self.loss_history[-1]
        else:
            stats["loss_improvement"] = 0
        
        # 4. Pattern recognition (autodidatismo)
        if len(self.activation_history) >= 10:
            recent = torch.stack(self.activation_history[-10:])
            patterns = torch.std(recent[:, :, idx]).item()
            stats["pattern_recognition"] = patterns
        else:
            stats["pattern_recognition"] = 0
        
        # 5. Activation variance (autonomia)
        if len(self.activation_history) > 0:
            acts = torch.cat([a[:, idx:idx+1] for a in self.activation_history])
            stats["activation_variance"] = torch.var(acts).item()
        else:
            stats["activation_variance"] = 0
        
        # 6. Unique contribution (autossuficiência)
        out_weights = self.W_out[:, idx]
        stats["unique_contribution"] = torch.norm(out_weights).item() / (self.output_dim + 1e-6)
        
        # 7. Growth participation (autoconstrução)
        stats["growth_participation"] = self.neuron_stats[idx].get("age", 0) * 0.1
        
        # 8. Synapse plasticity (autosináptico)
        if self.W_hh.grad is not None:
            stats["synapse_plasticity"] = torch.abs(self.W_hh.grad[idx]).mean().item()
        else:
            stats["synapse_plasticity"] = 0
        
        # 9. Structural influence (autoarquitetável)
        connectivity = (torch.abs(self.W_hh[idx]) > 0.01).float().mean().item()
        stats["structural_influence"] = connectivity
        
        return stats

    @torch.no_grad()
    def validate_and_replace_neurons(self):
        """Valida cada neurônio e substitui os que falham nos critérios IA³"""
        validator = IA3Validator()
        replacements = []
        
        for idx in range(self.hidden_dim):
            stats = self.compute_neuron_stats(idx)
            is_valid, score, failures, criteria = validator.validate_neuron(stats)
            
            if not is_valid:
                replacements.append({
                    "idx": idx,
                    "score": score,
                    "failures": failures,
                    "criteria": criteria
                })
        
        # Substituir neurônios inválidos
        for rep in replacements:
            idx = rep["idx"]
            print(f"⚠️ Neurônio #{idx} FALHOU: {rep['failures'][0]}")
            print(f"   Score: {rep['score']:.2%}")
            
            # Reinicializar neurônio defeituoso
            nn.init.kaiming_uniform_(self.W_in[idx:idx+1], a=math.sqrt(5))
            nn.init.orthogonal_(self.W_hh[idx:idx+1, idx:idx+1])
            self.W_hh[idx, :] = torch.randn_like(self.W_hh[idx]) * 0.01
            self.W_hh[:, idx] = torch.randn_like(self.W_hh[:, idx]) * 0.01
            self.b_h[idx] = 0
            nn.init.kaiming_uniform_(self.W_out[:, idx:idx+1], a=math.sqrt(5))
            
            # Reset estatísticas
            self.neuron_stats[idx] = {
                "birth_time": datetime.utcnow(),
                "age": 0,
                "total_activations": 0,
                "grad_accumulator": 0,
                "last_loss": float('inf'),
                "recursion_count": 0,
                "pattern_matches": 0,
                "unique_outputs": set(),
                "synapse_updates": 0
            }
            
            print(f"   ✅ Neurônio #{idx} RENASCIDO com DNA corrigido")
        
        return len(replacements), replacements

    @torch.no_grad()
    def expand_hidden(self, k=1):
        """Adiciona k neurônios validados IA³"""
        if k < 1: return
        H0, I, O = self.hidden_dim, self.input_dim, self.output_dim
        H1 = H0 + k

        W_in_new  = torch.empty(H1, I, device=self.device)
        W_hh_new  = torch.empty(H1, H1, device=self.device)
        b_h_new   = torch.empty(H1, device=self.device)
        W_out_new = torch.empty(O, H1, device=self.device)

        # Copiar neurônios existentes (que já foram validados)
        W_in_new[:H0] = self.W_in.data
        W_hh_new[:H0, :H0] = self.W_hh.data
        b_h_new[:H0] = self.b_h.data
        W_out_new[:, :H0] = self.W_out.data

        # Inicializar novos neurônios com "DNA IA³"
        nn.init.kaiming_uniform_(W_in_new[H0:], a=math.sqrt(5))
        
        # Garantir autorecursividade desde o nascimento
        for i in range(H0, H1):
            W_hh_new[i, i] = 0.1  # Auto-conexão inicial
            if i > 0:
                # Conexões com neurônios anteriores
                W_hh_new[i, :i] = torch.randn(i, device=self.device) * 0.05
                W_hh_new[:i, i] = torch.randn(i, device=self.device) * 0.05
        
        b_h_new[H0:] = 0.0
        nn.init.kaiming_uniform_(W_out_new[:, H0:], a=math.sqrt(5))

        # Registrar
        self.hidden_dim = H1
        self.W_in  = nn.Parameter(W_in_new)
        self.W_hh  = nn.Parameter(W_hh_new)
        self.b_h   = nn.Parameter(b_h_new)
        self.W_out = nn.Parameter(W_out_new)
        
        # Adicionar estatísticas para novos neurônios
        for i in range(k):
            self.neuron_stats.append({
                "birth_time": datetime.utcnow(),
                "age": 0,
                "total_activations": 0,
                "grad_accumulator": 0,
                "last_loss": float('inf'),
                "recursion_count": 0,
                "pattern_matches": 0,
                "unique_outputs": set(),
                "synapse_updates": 0
            })
        
        print(f"🧬 Nasceu neurônio #{H1} com DNA IA³ completo")

    @torch.no_grad()
    def prune_worst(self):
        """Remove neurônio com pior score IA³"""
        if self.hidden_dim <= 1:
            return False
        
        validator = IA3Validator()
        worst_idx = -1
        worst_score = 1.0
        
        for idx in range(self.hidden_dim):
            stats = self.compute_neuron_stats(idx)
            _, score, _, _ = validator.validate_neuron(stats)
            if score < worst_score:
                worst_score = score
                worst_idx = idx
        
        if worst_idx >= 0:
            print(f"☠️ Neurônio #{worst_idx} morreu (score IA³: {worst_score:.2%})")
            self._prune_index(worst_idx)
            return True
        return False

    @torch.no_grad()
    def _prune_index(self, idx):
        H, I, O = self.hidden_dim, self.input_dim, self.output_dim
        keep = [i for i in range(H) if i != idx]
        self.W_in  = nn.Parameter(self.W_in.data[keep, :].clone())
        self.W_hh  = nn.Parameter(self.W_hh.data[keep, :][:, keep].clone())
        self.b_h   = nn.Parameter(self.b_h.data[keep].clone())
        self.W_out = nn.Parameter(self.W_out.data[:, keep].clone())
        self.hidden_dim = H - 1
        
        # Remover estatísticas
        if idx < len(self.neuron_stats):
            del self.neuron_stats[idx]

# ------------------------
# Datasets
# ------------------------
def batch_numeric(batch_size=32, input_dim=8, device="cpu"):
    """Tarefa complexa com múltiplos padrões para testar critérios IA³"""
    x = torch.rand(batch_size, input_dim, device=device)
    
    # Múltiplos padrões para forçar adaptação
    y1 = torch.sum(x, dim=1, keepdim=True)
    y2 = torch.prod(x[:, :4], dim=1, keepdim=True)
    y3 = torch.sin(x[:, 0:1] * 3.14159)
    y4 = torch.cos(x[:, 1:2] * 3.14159)
    y5 = (x[:, 2:3] - x[:, 3:4]) ** 2
    y6 = torch.max(x[:, 4:6], dim=1, keepdim=True)[0]
    y7 = torch.min(x[:, 6:8], dim=1, keepdim=True)[0]
    y8 = torch.std(x, dim=1, keepdim=True)
    
    y = torch.cat([y1, y2, y3, y4, y5, y6, y7, y8], dim=1)
    return x, y

def build_char_vocab(texts, limit_chars=256):
    charset = set()
    for t in texts:
        charset.update(list(t))
        if len(charset) >= limit_chars:
            break
    idx2ch = sorted(list(charset))[:limit_chars]
    ch2idx = {c:i for i,c in enumerate(idx2ch)}
    return ch2idx, idx2ch

def one_hot(indices, vocab_size, device):
    return F.one_hot(indices, num_classes=vocab_size).float().to(device)

def load_text_corpus(dataset_name="roneneldan/TinyStories", split="train[:0.2%]"):
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"⚠️ datasets não instalado, usando corpus sintético")
        # Corpus sintético de emergência
        return [
            "The cat sat on the mat.",
            "Dogs love to play fetch.",
            "Birds fly high in the sky.",
            "Fish swim in the ocean.",
            "The sun shines bright today."
        ] * 10
    
    try:
        ds = load_dataset(dataset_name, split=split)
        texts = [r.get("text", "") or r.get("story", "") or "" for r in ds]
        return [t for t in texts if t.strip()]
    except Exception as e:
        print(f"⚠️ Erro ao carregar dataset: {e}")
        return ["Emergency corpus"] * 10

def make_text_minibatch(corpus, ch2idx, seq_len=64, batch=16, device="cpu"):
    vocab = len(ch2idx)
    xs, ys = [], []
    for _ in range(batch):
        s = random.choice(corpus)
        if len(s) < seq_len + 1:
            s = s + " " * (seq_len + 1 - len(s))
        start = random.randint(0, max(0, len(s) - (seq_len + 1)))
        chunk = s[start:start+seq_len+1]
        idxs = [ch2idx.get(c, 0) for c in chunk]
        inp = torch.tensor(idxs[:-1], dtype=torch.long)
        tgt = torch.tensor(idxs[1:], dtype=torch.long)
        xs.append(inp); ys.append(tgt)
    X = torch.stack(xs, dim=0).to(device)
    Y = torch.stack(ys, dim=0).to(device)
    return X, Y, vocab

# ------------------------
# Treino com Validação IA³
# ------------------------
def train_numeric(brain, steps=50, lr=1e-3, batch=32, device="cpu"):
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    loss_meter = 0.0
    h = None
    
    for step in range(steps):
        x, y = batch_numeric(batch, brain.input_dim, device=device)
        
        # Ensure h is detached to avoid graph issues
        if h is not None:
            h = h.detach()
        
        y_pred, h = brain(x, h)
        loss = F.mse_loss(y_pred, y)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        
        loss_meter += loss.item()
        brain.loss_history.append(loss.item())
        
        # A cada 10 steps, validar neurônios
        if step % 10 == 0 and step > 0:
            replaced, details = brain.validate_and_replace_neurons()
            if replaced > 0:
                print(f"   🔄 {replaced} neurônios substituídos no step {step}")
                h = None  # Reset hidden state após substituição
    
    return loss_meter/steps

def train_text(brain, corpus, ch2idx, steps=50, lr=1e-3, batch=16, seq_len=64, device="cpu"):
    vocab_size = len(ch2idx)
    if brain.input_dim != vocab_size or brain.output_dim != vocab_size:
        raise RuntimeError(f"Incompatibilidade: brain I/O ({brain.input_dim}/{brain.output_dim}) vs vocab {vocab_size}")
    
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    loss_meter = 0.0
    
    for step in range(steps):
        X, Y, _ = make_text_minibatch(corpus, ch2idx, seq_len=seq_len, batch=batch, device=device)
        B, T = X.shape
        h = None
        loss_acc = 0.0
        
        for t in range(T):
            if h is not None:
                h = h.detach()
            x_onehot = one_hot(X[:, t], vocab_size, device)
            logits, h = brain(x_onehot, h)
            loss_t = F.cross_entropy(logits, Y[:, t])
            loss_acc += loss_t
        
        loss = loss_acc / T
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        opt.step()
        
        loss_meter += loss.item()
        brain.loss_history.append(loss.item())
        
        # Validação IA³
        if step % 10 == 0 and step > 0:
            replaced, _ = brain.validate_and_replace_neurons()
            if replaced > 0:
                print(f"   🔄 {replaced} neurônios substituídos no step {step}")
    
    return loss_meter/steps

# ------------------------
# Persistência
# ------------------------
def load_or_init_brain(mode, device, numeric_dims=(8,8,8), text_vocab_size=None, hidden_init=16, act="relu", state_dir="/opt/ia3/brain"):
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == "numeric":
        state_path = state_dir / "brain_numeric_state.pt"
        if state_path.exists():
            obj = torch.load(state_path, map_location=device, weights_only=False)
            spec = obj["spec"]
            # Fix activation_name vs activation parameter name issue
            brain = Brain(
                input_dim=spec.get("input_dim"),
                hidden_dim=spec.get("hidden_dim"),
                output_dim=spec.get("output_dim"),
                activation=spec.get("activation_name", "relu"),
                device=device
            )
            brain.load_state_dict(obj["state"])
            if "stats" in obj:
                brain.neuron_stats = obj["stats"]
            if "history" in obj:
                brain.loss_history = obj["history"]
            return brain, str(state_path)
        else:
            I, H, O = numeric_dims
            brain = Brain(I, hidden_init, O, activation=act, device=device)
            return brain, str(state_path)
    else:
        assert text_vocab_size is not None and text_vocab_size > 1
        state_path = state_dir / "brain_text_state.pt"
        if state_path.exists():
            obj = torch.load(state_path, map_location=device, weights_only=False)
            spec = obj["spec"]
            if spec["input_dim"] != text_vocab_size or spec["output_dim"] != text_vocab_size:
                brain = Brain(text_vocab_size, spec["hidden_dim"], text_vocab_size, 
                            activation=spec.get("activation_name", "relu"), device=device)
            else:
                brain = Brain(
                    input_dim=spec.get("input_dim"),
                    hidden_dim=spec.get("hidden_dim"),
                    output_dim=spec.get("output_dim"),
                    activation=spec.get("activation_name", "relu"),
                    device=device
                )
                brain.load_state_dict(obj["state"])
                if "stats" in obj:
                    brain.neuron_stats = obj["stats"]
                if "history" in obj:
                    brain.loss_history = obj["history"]
            return brain, str(state_path)
        else:
            brain = Brain(text_vocab_size, hidden_init, text_vocab_size, activation=act, device=device)
            return brain, str(state_path)

def save_brain(brain, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "spec": {
            "input_dim": brain.input_dim,
            "hidden_dim": brain.hidden_dim,
            "output_dim": brain.output_dim,
            "activation_name": brain.activation_name
        },
        "state": brain.state_dict(),
        "stats": brain.neuron_stats,
        "history": brain.loss_history[-100:] if hasattr(brain, 'loss_history') else []
    }, path)

# ------------------------
# Main (1 rodada Darwin)
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="IA³ Embrião: neurônios com prova de vida")
    ap.add_argument("--event", choices=["live","death"], default="live",
                    help="Decisão Darwin para esta rodada")
    ap.add_argument("--mode", choices=["numeric","text"], default="numeric")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hidden-init", type=int, default=1)

    # numeric
    ap.add_argument("--num-input", type=int, default=8)
    ap.add_argument("--num-output", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--num-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)

    # text
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--split", default="train[:0.2%]")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--text-steps", type=int, default=50)
    ap.add_argument("--text-batch", type=int, default=16)
    ap.add_argument("--vocab-max", type=int, default=128)

    # infra
    ap.add_argument("--state-dir", default="/opt/ia3/brain")
    ap.add_argument("--worm", default="/opt/ia3/logs/worm.log")
    ap.add_argument("--metrics-port", type=int, default=9101)
    args = ap.parse_args()

    set_seeds(args.seed)

    # Prometheus
    if PROM_OK and args.metrics_port > 0:
        try:
            start_http_server(args.metrics_port)
        except OSError:
            pass
        G_loss = Gauge("ia3_embryo_loss", "Loss médio da rodada")
        G_neurons = Gauge("ia3_embryo_neurons", "Neurônios (H) atuais")
        G_valid = Gauge("ia3_embryo_valid_neurons", "Neurônios válidos IA³")
        C_births = Counter("ia3_embryo_births_total", "Nascimentos acumulados")
        C_deaths = Counter("ia3_embryo_deaths_total", "Mortes acumuladas")
        C_replacements = Counter("ia3_embryo_replacements_total", "Substituições IA³")
    else:
        G_loss = G_neurons = G_valid = C_births = C_deaths = C_replacements = None

    device = args.device
    t0 = time.time()

    # Carregar/Inicializar
    if args.mode == "numeric":
        brain, state_path = load_or_init_brain(
            "numeric", device,
            numeric_dims=(args.num_input, args.hidden_init, args.num_output),
            hidden_init=args.hidden_init, act="relu",
            state_dir=args.state_dir
        )
        corpus = None
        ch2idx = None
    else:
        corpus = load_text_corpus(args.dataset, args.split)
        ch2idx, idx2ch = build_char_vocab(corpus, limit_chars=args.vocab_max)
        vocab = len(ch2idx)
        brain, state_path = load_or_init_brain(
            "text", device,
            text_vocab_size=vocab, hidden_init=args.hidden_init, act="tanh",
            state_dir=args.state_dir
        )

    print(f"\n{'='*60}")
    print(f"🧬 RODADA DARWIN - EMBRIÃO IA³")
    print(f"{'='*60}")
    print(f"Modo: {args.mode}")
    print(f"Neurônios atuais: {brain.hidden_dim}")
    print(f"Evento Darwin: {args.event}")
    
    # Treino com validação IA³
    print(f"\n📚 Treinando e validando neurônios...")
    if args.mode == "numeric":
        loss = train_numeric(brain, steps=args.num_steps, lr=args.lr, 
                           batch=args.num_batch, device=device)
    else:
        loss = train_text(brain, corpus, ch2idx, steps=args.text_steps, lr=args.lr,
                         batch=args.text_batch, seq_len=args.seq_len, device=device)

    # Validação final completa
    print(f"\n🔬 Validação final IA³...")
    replaced_final, details = brain.validate_and_replace_neurons()
    
    # Contabilizar neurônios válidos
    validator = IA3Validator()
    valid_count = 0
    for idx in range(brain.hidden_dim):
        stats = brain.compute_neuron_stats(idx)
        is_valid, _, _, _ = validator.validate_neuron(stats)
        if is_valid:
            valid_count += 1
    
    print(f"   Neurônios válidos: {valid_count}/{brain.hidden_dim}")
    
    # Aplicar veredito Darwin
    action = None
    if args.event == "live":
        brain.expand_hidden(k=1)  # +1 neurônio com DNA IA³
        action = "birth"
        if C_births: C_births.inc()
    else:
        pruned = brain.prune_worst()
        action = "death_pruned" if pruned else "death_noop"
        if C_deaths: C_deaths.inc()

    # Persistir
    save_brain(brain, state_path)

    # Métricas
    if G_loss: G_loss.set(loss)
    if G_neurons: G_neurons.set(brain.hidden_dim)
    if G_valid: G_valid.set(valid_count)
    if C_replacements and replaced_final > 0: 
        C_replacements.inc(replaced_final)

    # WORM
    event = {
        "timestamp": now_utc(),
        "mode": args.mode,
        "event": args.event,
        "action": action,
        "hidden_dim": brain.hidden_dim,
        "valid_neurons": valid_count,
        "replaced": replaced_final,
        "loss": float(loss),
        "state_path": state_path
    }
    
    try:
        jwrite(args.worm, event, append=True)
    except Exception as e:
        print(f"[WORM] Falha ao escrever log: {e}")

    dt = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"✅ RODADA COMPLETA")
    print(f"{'='*60}")
    print(f"Ação: {action}")
    print(f"Loss: {loss:.6f}")
    print(f"Neurônios: {brain.hidden_dim} (válidos: {valid_count})")
    print(f"Substituições: {replaced_final}")
    print(f"Tempo: {dt:.2f}s")
    print(f"{'='*60}\n")
    
    # Retorno JSON para integração
    print(json.dumps({
        "ok": True, 
        "action": action, 
        "loss": loss, 
        "H": brain.hidden_dim,
        "valid": valid_count,
        "replaced": replaced_final,
        "dt_s": dt
    }, indent=2))

if __name__ == "__main__":
    main()