# Arquitetura — Núcleo Imutável, Periferia Plugável

[Módulos de Domínio]  -->  (E,N,I)  -->  [ infty-bar núcleo ]  -->  P  -->  [Consumidores]
   (LLM, RL, Q, MA)          ^                               (logs, métricas, ações)
                           auditoria

- Quântico (Q): gera N_q e I_q e compõe com N/I totais.
- Multiagente (MA): agrega E/N de agentes; I coletivo via min/produto/ponderado.
- BioIA: mutações fornecem N_bio + admissibilidade; infty-bar decide inclusão.

Contratos:
- Módulos retornam contribuições (E,N) e métricas I_i.
- Agregador compõe E_total, N_total, I_total e chama infty-bar.
- Núcleo não pode ser modificado.
