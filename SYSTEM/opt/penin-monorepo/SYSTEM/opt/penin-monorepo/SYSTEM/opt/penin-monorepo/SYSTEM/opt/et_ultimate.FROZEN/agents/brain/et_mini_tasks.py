# /opt/et_ultimate/agents/brain/et_mini_tasks.py
"""
Gerador determinístico de mini-tarefas para o ciclo ETΩ.

Exporta:
- gerar_mini_tarefas(base_equation: str, n: int = 5, seeds: list[str] | None = None) -> list[dict]
  Retorna uma lista de tarefas pequenas e independentes, usadas para guiar
  agentes de mutação/avaliação.

Design:
- Sem dependências externas
- Não executa nada no import
- Saída estável e serializável (dicts simples)
"""

from __future__ import annotations
import time
import uuid
from typing import List, Dict, Optional


def _mk_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4()}"


def gerar_mini_tarefas(
    base_equation: str,
    n: int = 5,
    seeds: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Gera um conjunto de mini-tarefas para orientar mutações e avaliações.

    Parâmetros:
      - base_equation: equação base atual (string)
      - n: número desejado de mini-tarefas (default: 5)
      - seeds: pistas/temas opcionais para diversificar as tarefas

    Retorno:
      - lista de dicionários com chaves:
          id, ts, kind, prompt, meta
    """
    if not isinstance(base_equation, str) or not base_equation.strip():
        base_equation = "x^2 + y^2"

    n = max(1, int(n))
    seeds = seeds or []

    tasks: List[Dict] = []
    ts = time.time()

    # 1) Tarefas de mutação (busca local/global)
    templates_mut = [
        "Produza uma mutação conservadora que simplifique sem perder capacidade: {eq}",
        "Proponha uma mutação agressiva que maximize melhoria esperada: {eq}",
        "Reescreva com foco em estabilidade numérica e termos bem condicionados: {eq}",
        "Introduza regularização mínima para reduzir overfit estrutural: {eq}",
        "Aplique fator de entropia controlada para evitar colapso: {eq}",
        "Avalie e substitua operadores redundantes por equivalentes mais simples: {eq}",
    ]

    # 2) Tarefas de avaliação (explicabilidade/checagem rápida)
    templates_eval = [
        "Explique em 3 linhas por que a mutação proposta tende a generalizar melhor que: {eq}",
        "Liste 3 riscos numéricos potenciais da mutação vs base: {eq}",
        "Indique 2 cenários adversários em que a mutação pode falhar: {eq}",
    ]

    # Mistura seeds simples como variações de foco
    seed_prompts = [f"[foco:{s.strip()}] " for s in seeds if isinstance(s, str) and s.strip()]

    # Geração determinística simples: intercalar mutação/avaliação até n
    i_mut, i_eval, i_seed = 0, 0, 0
    while len(tasks) < n:
        if len(tasks) % 2 == 0:
            # mutação
            tmpl = templates_mut[i_mut % len(templates_mut)]
            seed = seed_prompts[i_seed % len(seed_prompts)] if seed_prompts else ""
            prompt = seed + tmpl.format(eq=base_equation)
            tasks.append({
                "id": _mk_id("mut"),
                "ts": ts,
                "kind": "mutate",
                "prompt": prompt,
                "meta": {
                    "base": base_equation,
                    "strategy": "local" if (i_mut % 2 == 0) else "global",
                },
            })
            i_mut += 1
            i_seed += 1
        else:
            # avaliação
            tmpl = templates_eval[i_eval % len(templates_eval)]
            seed = seed_prompts[i_seed % len(seed_prompts)] if seed_prompts else ""
            prompt = seed + tmpl.format(eq=base_equation)
            tasks.append({
                "id": _mk_id("eval"),
                "ts": ts,
                "kind": "evaluate",
                "prompt": prompt,
                "meta": {
                    "base": base_equation,
                    "criteria": "explicabilidade+robustez",
                },
            })
            i_eval += 1
            i_seed += 1

    return tasks


if __name__ == "__main__":
    # Teste rápido (não roda no import)
    demo = gerar_mini_tarefas("x^2 + y^2 + λ·xy", n=6, seeds=["estabilidade", "simplicidade"])
    from json import dumps
    print(dumps(demo, ensure_ascii=False, indent=2))
