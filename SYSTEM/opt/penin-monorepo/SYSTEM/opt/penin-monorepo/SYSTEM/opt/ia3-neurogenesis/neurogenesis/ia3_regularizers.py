import torch
from typing import Dict

class EWC:
    """Elastic Weight Consolidation (aprox. Fisher diagonal) — protege conhecimento entre ciclos."""
    def __init__(self, model, lam: float = 2.0):
        self.model = model
        self.lam = lam
        self.fisher: Dict[str, torch.Tensor] = {}
        self.star: Dict[str, torch.Tensor] = {}

    def consolidate(self, batch):
        self.model.zero_grad(set_to_none=True)
        X, Y = batch
        pred = self.model(X)
        loss = torch.nn.functional.mse_loss(pred, Y)
        loss.backward()

        # Fisher diagonal ~ (grad^2)
        for n, p in self.model.named_parameters():
            if p.grad is None: 
                continue
            g2 = p.grad.detach()**2
            self.fisher[n] = g2.clone() if n not in self.fisher else 0.9*self.fisher[n] + 0.1*g2
            self.star[n] = p.detach().clone()

    def penalty(self, model):
        if not self.fisher:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        reg = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                reg = reg + (self.fisher[n] * (p - self.star[n])**2).sum()
        return self.lam * reg

class MAS:
    """Memory Aware Synapses — importância a partir da sensibilidade da saída."""
    def __init__(self, model, lam: float = 1.0):
        self.model = model
        self.lam = lam
        self.omega: Dict[str, torch.Tensor] = {}
        self.star: Dict[str, torch.Tensor] = {}

    def consolidate(self, batch):
        self.model.zero_grad(set_to_none=True)
        X, Y = batch
        out = self.model(X)
        # ||f(x)||_2 sensibilidade
        s = out.norm(2)
        s.backward()
        for n, p in self.model.named_parameters():
            if p.grad is None: 
                continue
            contrib = p.grad.detach().abs()
            self.omega[n] = contrib.clone() if n not in self.omega else 0.9*self.omega[n] + 0.1*contrib
            self.star[n] = p.detach().clone()

    def penalty(self, model):
        if not self.omega:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        reg = 0.0
        for n, p in model.named_parameters():
            if n in self.omega:
                reg = reg + (self.omega[n] * (p - self.star[n]).abs()).sum()
        return self.lam * reg