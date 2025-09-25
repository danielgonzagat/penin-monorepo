# Instruções — Upload do Mac para o Servidor
No Mac (Terminal):
```bash
scp ~/Downloads/ETOmega_corpus_cleaned.zip root@SEU_IP:/opt/et_fusion/data/
```

No servidor (SSH):
```bash
ssh root@SEU_IP
mkdir -p /opt/et_fusion/data/corpus/turing_clean
cd /opt/et_fusion/data
unzip -o ETOmega_corpus_cleaned.zip -d /opt/et_fusion/data/corpus/turing_clean
ETKB_CORPUS=/opt/et_fusion/data/corpus/turing_clean ETKB_INDEX=/opt/et_fusion/data/index python3 /opt/et_fusion/router/etkb.py
```
