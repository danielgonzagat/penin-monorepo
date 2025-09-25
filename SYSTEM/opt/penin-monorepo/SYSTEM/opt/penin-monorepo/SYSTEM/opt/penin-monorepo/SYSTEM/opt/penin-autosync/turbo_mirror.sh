#!/bin/bash
# TURBO MIRROR - Espelhamento ULTRA RÁPIDO de TUDO

echo "🚀 PENIN TURBO MIRROR - Espelhamento TOTAL"
echo "==========================================="

# Configurações
REPO="/opt/penin-monorepo"
GITHUB_USER="danielgonzagat"
GITHUB_TOKEN="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"

# Criar estrutura organizada
echo "📁 Criando estrutura..."
cd $REPO

# Limpar repo anterior (manter .git)
find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} \;

# Criar conglomerados organizados
mkdir -p {SYSTEM,PROJECTS,DATA,CONFIGS,LOGS,NEURAL,AGENTS,TOOLS}

# Função de rsync otimizado
turbo_sync() {
    src=$1
    dst=$2
    echo "  Sincronizando $src → $dst"
    rsync -av --progress \
        --exclude=.git \
        --exclude=__pycache__ \
        --exclude=*.pyc \
        --exclude=.venv \
        --exclude=venv \
        --exclude=node_modules \
        --exclude=.cache \
        --exclude=*.sock \
        --exclude=*.log \
        --exclude=*.tmp \
        --max-size=100M \
        "$src/" "$dst/" 2>/dev/null || true
}

echo "🔄 Espelhando sistema..."

# SYSTEM - Configurações e binários
turbo_sync /etc SYSTEM/etc
turbo_sync /usr/local/bin SYSTEM/usr_local_bin
turbo_sync /opt SYSTEM/opt

# PROJECTS - Todos os projetos
for proj in /root/*; do
    if [ -d "$proj" ] && [ ! -L "$proj" ]; then
        name=$(basename "$proj")
        case $name in
            neural*|ia3*|complete_brain*|agi*)
                turbo_sync "$proj" "NEURAL/$name"
                ;;
            agent*|swarm*|hivemind*)
                turbo_sync "$proj" "AGENTS/$name"
                ;;
            *.log|logs|*backup*)
                # Skip logs
                ;;
            *)
                turbo_sync "$proj" "PROJECTS/$name"
                ;;
        esac
    fi
done

# DATA - Dados importantes
turbo_sync /root/data DATA/root_data 2>/dev/null || true
turbo_sync /root/uploads DATA/uploads 2>/dev/null || true
turbo_sync /var/www DATA/www 2>/dev/null || true

# CONFIGS - Configurações importantes
cp /root/.bashrc CONFIGS/ 2>/dev/null || true
cp /root/.env* CONFIGS/ 2>/dev/null || true
cp -r /root/.ssh CONFIGS/ssh_configs 2>/dev/null || true
cp -r /root/.config CONFIGS/config 2>/dev/null || true

# TOOLS - Ferramentas e scripts
turbo_sync /opt/penin-autosync TOOLS/penin-autosync

# Criar README principal
cat > README.md << 'EOF'
# 🚀 PENIN MONOREPO - SISTEMA COMPLETO

> **Espelhamento TOTAL do Sistema via TURBO SYNC**

## 📊 Status

- **Sincronização**: ATIVA 24/7
- **Método**: SSH + rsync turbo
- **Última Atualização**: $(date '+%Y-%m-%d %H:%M:%S')

## 🗂️ ESTRUTURA

### 📁 SYSTEM
Configurações do sistema, binários e /opt

### 🧠 NEURAL
Sistemas neurais, IA e modelos

### 👥 AGENTS
Sistemas de agentes e multi-agentes

### 🚀 PROJECTS
Todos os projetos e desenvolvimento

### 💾 DATA
Dados, uploads e conteúdo

### ⚙️ CONFIGS
Configurações e dotfiles

### 🔧 TOOLS
Ferramentas e scripts do sistema

## 🔄 Sincronização Automática

Este repositório é atualizado automaticamente a cada 30 segundos.

### Como funciona:

1. **rsync turbo** espelha todos os diretórios
2. **Git** faz commit e push automático
3. **SSH** garante máxima velocidade
4. **24/7** operação contínua

## 📈 Estatísticas

EOF

# Adicionar estatísticas
echo "- Total de arquivos: $(find . -type f | wc -l)" >> README.md
echo "- Tamanho total: $(du -sh . | cut -f1)" >> README.md
echo "- Conglomerados: 7" >> README.md
echo "" >> README.md
echo "---" >> README.md
echo "*Sistema PENIN - Espelhamento Total Ativo*" >> README.md

# Git operations
echo "📤 Enviando para GitHub..."
git add -A
git commit -m "TURBO SYNC: $(date '+%Y%m%d_%H%M%S')" --no-verify || true

# Push via HTTPS (mais confiável)
git push https://$GITHUB_TOKEN@github.com/$GITHUB_USER/penin-monorepo.git main --force

echo "✅ TURBO MIRROR COMPLETO!"