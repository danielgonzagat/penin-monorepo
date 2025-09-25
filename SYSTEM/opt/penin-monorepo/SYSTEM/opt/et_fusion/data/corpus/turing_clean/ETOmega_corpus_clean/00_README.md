# ETΩ — Corpus Canônico (Limpo)
Este pacote foi reorganizado para facilitar compreensão por LLMs.

## Estrutura
- `01_Canon/01_MarcoZero/` — documento origem (marco zero).
- `01_Canon/02_Evolucao_LinhaDoTempo/` — evolução numerada (10,20,...).
- `01_Canon/99_BestAtual/` — versão canônica atual.
- `02_Support/References/` — materiais de apoio.
- `03_Code/` — código.
- `04_Data/` — dados.
- `05_Appendix/Archives/` — anexos/compactados.
- `99_Index/` — mapeamentos e duplicatas removidas.

## Manifesto
Veja `00_manifest.json` para `marco_zero`, `best_atual` e `ordem_evolucao`.

## Uso com RAG local
1. Copie esta pasta para `/opt/et_fusion/data/corpus/turing_clean`
2. Reindexe:
   ```bash
   ETKB_CORPUS=/opt/et_fusion/data/corpus/turing_clean \
   ETKB_INDEX=/opt/et_fusion/data/index \
   python3 /opt/et_fusion/router/etkb.py
   ```

## Observações
- Duplicatas **exatas** foram removidas (ver `99_Index/duplicates_removed.tsv`).
- Para near-duplicates, rode uma limpeza semântica posterior.
