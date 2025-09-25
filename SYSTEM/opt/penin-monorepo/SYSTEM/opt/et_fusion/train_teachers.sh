#!/bin/bash

# Endereço da API do ensemble router
API_URL="http://127.0.0.1:8800/v1/chat/completions"

# Caminho dos textos extraídos
CORPUS_DIR="/opt/et_fusion/data/lemniscata_corpus_txt"

# Parâmetros para as chamadas (ajuste se quiser mais/menos tokens)
TEMPERATURE=1.0
MAX_TOKENS=1024
TOP_K_CTX=6

for txt in "$CORPUS_DIR"/*.txt; do
    # Pega os primeiros 2000 caracteres do documento (para não exceder o limite de entrada)
    content=$(head -c 2000 "$txt" | tr '\n' ' ')
    doc=$(basename "$txt")
    prompt="Leia o seguinte documento sobre a história e evolução da Equação de Turing até a Lemniscata de Penin e resuma seus pontos principais em português. Documento: $content"

    # Monta o JSON via jq (evita problemas com aspas)
    json=$(jq -nc --arg p "$prompt" --arg m "ensemble-5x" --argjson t $TEMPERATURE --argjson max $MAX_TOKENS --argjson k $TOP_K_CTX '
    {
      model: $m,
      messages: [{role:"user", content:$p}],
      temperature: $t,
      max_tokens: $max,
      top_k_ctx: $k
    }')

    # Chama o router e exibe a resposta na tela
    echo ">> Treinando com $doc"
    curl -s "$API_URL" -H 'Content-Type: application/json' -d "$json" | jq -r '.choices[0].message.content'

    # Pequena pausa para não saturar
    sleep 2
done
echo "Treino concluído!"
