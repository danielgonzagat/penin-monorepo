#!/bin/bash
# Sistema de Instalação Automática PENIN
# Script completo de instalação e configuração

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para imprimir status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Header
echo "================================================"
echo "   PENIN - Sistema de Sincronização Automática"
echo "   CPU → GitHub com README Mestre"
echo "================================================"
echo ""

# Verificar se está rodando como root
if [ "$EUID" -ne 0 ]; then 
    print_error "Por favor, execute como root (sudo)"
    exit 1
fi

print_status "Iniciando instalação do sistema PENIN..."

# Criar diretórios necessários
print_info "Criando estrutura de diretórios..."
mkdir -p /opt/penin-autosync/{tools,scripts,config,docs/sections,services/systemd,logs}
mkdir -p /opt/penin-monorepo
mkdir -p /opt/et_ultimate/agents/brain
mkdir -p /opt/ml/models
mkdir -p /opt/penin_omega/core
mkdir -p /root/projetos

print_status "Diretórios criados com sucesso"

# Instalar dependências do sistema
print_info "Instalando dependências do sistema..."
apt-get update -qq
apt-get install -y -qq git git-lfs python3 python3-pip python3-venv rsync inotify-tools curl wget > /dev/null 2>&1

print_status "Dependências do sistema instaladas"

# Configurar Git LFS
print_info "Configurando Git LFS..."
git lfs install

print_status "Git LFS configurado"

# Criar ambiente virtual Python
print_info "Criando ambiente virtual Python..."
if [ -d "/opt/penin-autosync/venv" ]; then
    rm -rf /opt/penin-autosync/venv
fi
python3 -m venv /opt/penin-autosync/venv

print_status "Ambiente virtual criado"

# Instalar dependências Python
print_info "Instalando dependências Python..."
/opt/penin-autosync/venv/bin/pip install --upgrade pip > /dev/null 2>&1
/opt/penin-autosync/venv/bin/pip install \
    watchdog \
    pyyaml \
    jinja2 \
    requests \
    gitpython > /dev/null 2>&1

print_status "Dependências Python instaladas"

# Criar arquivo de requisitos
cat > /opt/penin-autosync/requirements.txt << 'EOF'
watchdog>=3.0.0
pyyaml>=6.0
jinja2>=3.0.0
requests>=2.28.0
gitpython>=3.1.0
EOF

print_status "Arquivo de requisitos criado"

# Configurar permissões
chown -R $SUDO_USER:$SUDO_USER /opt/penin-* 2>/dev/null || chown -R root:root /opt/penin-*

print_status "Permissões configuradas"

# Criar script de execução
cat > /opt/penin-autosync/start.sh << 'EOF'
#!/bin/bash
# Script de inicialização do sistema PENIN

# Ativar ambiente virtual
source /opt/penin-autosync/venv/bin/activate

# Carregar variáveis de ambiente
if [ -f /opt/penin-autosync/.env ]; then
    source /opt/penin-autosync/.env
fi

# Executar sincronizador
exec python /opt/penin-autosync/tools/auto_sync.py /opt/penin-autosync/config/config.yaml
EOF

chmod +x /opt/penin-autosync/start.sh

print_status "Script de execução criado"

# Verificar configuração do GitHub
print_info "Verificando configuração do GitHub..."
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    print_status "Conexão com GitHub configurada corretamente"
else
    print_error "Conexão com GitHub não configurada. Configure sua chave SSH."
fi

echo ""
echo "================================================"
echo "           Instalação Concluída!"
echo "================================================"
echo ""
echo "Próximos passos:"
echo ""
echo "1. Configure sua chave do Cursor API:"
echo "   export CURSOR_API_KEY='sua_chave_aqui'"
echo ""
echo "2. Edite o arquivo de configuração:"
echo "   nano /opt/penin-autosync/config/config.yaml"
echo ""
echo "3. Execute o sistema:"
echo "   /opt/penin-autosync/start.sh"
echo ""
echo "Ou configure como serviço systemd:"
echo "   sudo systemctl enable penin-sync.service"
echo "   sudo systemctl start penin-sync.service"
echo ""
echo "================================================"