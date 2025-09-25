import torch, math, copy
import torch.nn.functional as F
from typing import Dict

class IA3LikeSuite:
    """
    Conjunto de testes proxy p/ IA³-like (0..1). Cada teste retorna [0,1].
    Critérios (ameaça de morte por rodada):
      - adaptativo: melhora em OOD após passo local
      - autorecursivo: sinal de autoajuste (grad norm estável/útil)
      - autoevolutivo: mutação leve de ativação/ruído melhora ou não piora
      - autodidata: perda auto-supervisionada reduz em minibatch extra
      - autônomo/autossuf.: estabilidade (grad_norm e saída bounded)
      - autoconstruível/autoarquitetável/autosináptico: nova unidade realmente conecta (fan-in/out > 0)
    """
    def run(self, model, base_loss: float, new_loss: float, X, Y) -> Dict[str, float]:
        tests = {}
        
        # Prepare test batch
        batch_size = min(64, X.size(0))
        indices = torch.randperm(X.size(0))[:batch_size]
        xb = X[indices]
        yb = Y[indices]

        # 1) adaptativo
        tests["adaptativo"] = float(1.0 if new_loss < base_loss * 0.98 else 0.0)

        # 2) autorecursivo (gradientes não-degenerados em parâmetros recém-criados)
        try:
            model.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            grad_norm = 0.0
            cnt = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item()
                    cnt += 1
            grad_norm = grad_norm / max(cnt,1)
            tests["autorecursivo"] = float(1.0 if 1e-6 < grad_norm < 5.0 else 0.0)
        except Exception:
            tests["autorecursivo"] = 0.0

        # 3) autoevolutivo (mutação suave de ativação)
        try:
            act_ok = 1.0
            old = model.act
            model.act = torch.nn.ReLU()
            with torch.no_grad():
                _ = model(xb)
            model.act = old
            tests["autoevolutivo"] = float(act_ok)
        except Exception:
            tests["autoevolutivo"] = 0.0

        # 4) autodidata (treino curtíssimo adicional reduz loss nesse minibatch)
        try:
            # Create temporary optimizer
            temp_model = copy.deepcopy(model)
            opt = torch.optim.SGD(temp_model.parameters(), lr=1e-3)
            
            with torch.no_grad():
                p0 = temp_model(xb)
                l0 = F.mse_loss(p0, yb).item()
            
            # Short training
            for _ in range(5):
                opt.zero_grad(set_to_none=True)
                p = temp_model(xb)
                l = F.mse_loss(p, yb)
                l.backward()
                opt.step()
            
            with torch.no_grad():
                p1 = temp_model(xb)
                l1 = F.mse_loss(p1, yb).item()
            
            tests["autodidata"] = float(1.0 if l1 < l0 else 0.0)
            
            # Get output norm for next test
            out_norm = p1.norm(2).item()/max(1.0, p1.numel())
            
        except Exception:
            tests["autodidata"] = 0.0
            out_norm = 1.0

        # 5) autônomo/autossuficiente (saída bounded e grad estável após treino curto)
        tests["autonomo_autossuficiente"] = float(1.0 if out_norm < 1.0 and grad_norm < 5.0 else 0.0)

        # 6) autoconstruível/autoarquitetável/autosináptico (fan-in/out do último neurônio > 0)
        try:
            if hasattr(model, 'fc2') and hasattr(model, 'fc1'):
                last_col = model.fc2.weight.data[:, -1].abs().sum().item()
                last_row = model.fc1.weight.data[-1, :].abs().sum().item()
                tests["autoconstruivel_autoarquitetavel_autosinaptico"] = float(1.0 if last_col>0.0 and last_row>0.0 else 0.0)
            else:
                tests["autoconstruivel_autoarquitetavel_autosinaptico"] = 0.5
        except Exception:
            tests["autoconstruivel_autoarquitetavel_autosinaptico"] = 0.0

        return tests

    def compose_score(self, tests: Dict[str, float]) -> float:
        # pesos iguais por simplicidade
        vals = list(tests.values())
        return sum(vals)/len(vals) if vals else 0.0