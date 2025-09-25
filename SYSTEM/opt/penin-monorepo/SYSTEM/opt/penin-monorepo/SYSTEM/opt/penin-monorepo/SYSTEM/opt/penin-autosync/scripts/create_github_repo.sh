#!/bin/bash
# Script para criar repositório no GitHub via API

echo "================================================"
echo "   Criar Repositório GitHub - PENIN Monorepo"
echo "================================================"
echo ""

# Verificar se gh está instalado
if ! command -v gh &> /dev/null; then
    echo "Instalando GitHub CLI..."
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update
    sudo apt install gh -y
fi

# Verificar autenticação
if ! gh auth status &> /dev/null; then
    echo "Por favor, autentique-se no GitHub:"
    gh auth login
fi

# Criar repositório
echo "Criando repositório 'penin-monorepo'..."
gh repo create penin-monorepo \
    --public \
    --description "Sistema PENIN - Evolução Contínua do Zero ao State-of-the-Art" \
    --clone=false \
    --confirm

if [ $? -eq 0 ]; then
    echo "✓ Repositório criado com sucesso!"
    
    # Fazer push inicial
    cd /opt/penin-monorepo
    git remote remove origin 2>/dev/null
    git remote add origin git@github.com:$(gh api user --jq .login)/penin-monorepo.git
    git branch -M main
    git push -u origin main
    
    echo "✓ Push inicial realizado!"
    echo ""
    echo "Acesse seu repositório em:"
    echo "https://github.com/$(gh api user --jq .login)/penin-monorepo"
else
    echo "✗ Erro ao criar repositório"
    echo "Você pode criar manualmente em: https://github.com/new"
fi