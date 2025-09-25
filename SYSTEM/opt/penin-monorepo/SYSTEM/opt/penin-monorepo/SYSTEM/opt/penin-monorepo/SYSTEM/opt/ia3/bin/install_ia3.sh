#!/bin/bash
# /opt/ia3/bin/install_ia3.sh
# Script de instalação completa do Sistema Celular IA³

set -euo pipefail

echo "🧬 INSTALANDO SISTEMA CELULAR IA³ NEUROEVOLUTIVO"
echo "=" * 60

# 1. Verificar dependências Python
echo "📦 Verificando dependências Python..."

REQUIRED_PACKAGES=(
    "torch"
    "prometheus_client"
)

OPTIONAL_PACKAGES=(
    "mlflow"
    "datasets"
    "tokenizers"
)

# Instalar dependências obrigatórias
for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "   ✅ $package já instalado"
    else
        echo "   📥 Instalando $package..."
        pip install $package
    fi
done

# Instalar dependências opcionais
for package in "${OPTIONAL_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "   ✅ $package já instalado"
    else
        echo "   📥 Instalando $package (opcional)..."
        pip install $package || echo "   ⚠️ Falha ao instalar $package (continuando...)"
    fi
done

# 2. Configurar permissões
echo "🔐 Configurando permissões..."
chown -R $USER:$USER /opt/ia3
find /opt/ia3 -name "*.py" -exec chmod +x {} \;
find /opt/ia3 -name "*.sh" -exec chmod +x {} \;
chmod +x /opt/ia3/bin/darwinctl

# 3. Criar symlinks para facilitar acesso
echo "🔗 Criando symlinks..."
ln -sf /opt/ia3/bin/darwinctl /usr/local/bin/darwinctl
ln -sf /opt/ia3/bin/ia3_neurogenesis.py /usr/local/bin/ia3-neurogenesis

# 4. Instalar serviço systemd
echo "🛠️ Instalando serviço systemd..."
cp /opt/ia3/services/ia3-neurogenesis.service /etc/systemd/system/
systemctl daemon-reload

echo "✅ INSTALAÇÃO COMPLETA!"
echo ""
echo "🚀 PRÓXIMOS PASSOS:"
echo "   1. Iniciar sistema: darwinctl start"
echo "   2. Ver status: darwinctl status"
echo "   3. Monitorar: darwinctl metrics --watch"
echo "   4. Ver logs: darwinctl worm -f"
echo ""
echo "📊 MÉTRICAS PROMETHEUS:"
echo "   URL: http://localhost:9093/metrics"
echo ""
echo "📈 DASHBOARD GRAFANA:"
echo "   Importar: /opt/ia3/dashboards/ia3_dashboard.json"
echo ""
echo "🔧 CONFIGURAÇÃO AVANÇADA:"
echo "   Editar: /opt/ia3/services/ia3-neurogenesis.service"
echo "   Habilitar systemd: systemctl enable ia3-neurogenesis"