# 🛠️ Guia de Configuração - Do Zero ao Sistema Completo

## Pré-requisitos do Sistema

### Requisitos Mínimos
- **Sistema Operacional**: Ubuntu 20.04+ ou equivalente
- **RAM**: 8GB mínimo (16GB recomendado)
- **Armazenamento**: 50GB livre
- **CPU**: 4 cores mínimo (8 cores recomendado)
- **GPU**: Opcional, mas recomendada para ML (NVIDIA com CUDA)

### Software Necessário
```bash
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependências básicas
sudo apt install -y \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
    wget \
    rsync \
    inotify-tools \
    htop \
    tree \
    vim \
    nano
```

## Configuração Inicial do Git

### 1. Configurar Identidade
```bash
git config --global user.name "Daniel Penin"
git config --global user.email "seu-email@exemplo.com"
```

### 2. Configurar SSH para GitHub
```bash
# Gerar nova chave SSH (se não tiver)
ssh-keygen -t ed25519 -C "seu-email@exemplo.com"

# Adicionar chave ao ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copiar chave pública
cat ~/.ssh/id_ed25519.pub
```

**Adicione a chave pública ao GitHub:**
1. Acesse GitHub → Settings → SSH and GPG keys
2. Clique em "New SSH key"
3. Cole a chave pública
4. Teste a conexão:
```bash
ssh -T git@github.com
```

## Instalação do Sistema PENIN

### 1. Criar Estrutura de Diretórios
```bash
# Criar diretórios principais
sudo mkdir -p /opt/penin-autosync/{tools,scripts,config,docs/sections,services/systemd}
sudo mkdir -p /opt/penin-monorepo
sudo mkdir -p /opt/et_ultimate
sudo mkdir -p /opt/ml
sudo mkdir -p /opt/penin_omega

# Definir permissões
sudo chown -R $USER:$USER /opt/penin-*
```

### 2. Instalar Dependências Python
```bash
# Criar ambiente virtual
python3 -m venv /opt/penin-autosync/venv
source /opt/penin-autosync/venv/bin/activate

# Instalar dependências
pip install --upgrade pip
pip install \
    watchdog \
    pyyaml \
    jinja2 \
    requests \
    gitpython \
    fastapi \
    uvicorn \
    sqlalchemy \
    redis \
    torch \
    transformers \
    scikit-learn \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    jupyter \
    pytest \
    black \
    isort \
    flake8 \
    mypy
```

### 3. Configurar Variáveis de Ambiente
```bash
# Criar arquivo de ambiente
cat > /opt/penin-autosync/.env << 'EOF'
# Configurações do Sistema PENIN
PENIN_ROOT=/opt/penin-autosync
PENIN_REPO=/opt/penin-monorepo
PENIN_CONFIG=/opt/penin-autosync/config/config.yaml

# GitHub
GITHUB_USER=danielgonzagat
GITHUB_REPO=penin-monorepo

# Cursor API (configure sua chave)
CURSOR_API_KEY=sua_chave_api_cursor_aqui

# Configurações de Logging
LOG_LEVEL=INFO
LOG_FILE=/opt/penin-autosync/logs/sync.log

# Configurações de Performance
MAX_WORKERS=4
COALESCE_SECONDS=0
DEBOUNCE_SECONDS=5
EOF

# Carregar variáveis
source /opt/penin-autosync/.env
```

## Configuração do Repositório GitHub

### 1. Criar Repositório
1. Acesse GitHub e crie um novo repositório
2. Nome: `penin-monorepo`
3. Descrição: "Sistema PENIN - Evolução Contínua do Zero ao SOTA"
4. Público ou Privado (sua escolha)
5. **NÃO** inicializar com README (será gerado automaticamente)

### 2. Configurar Repositório Local
```bash
cd /opt/penin-monorepo

# Inicializar Git
git init

# Configurar remote
git remote add origin git@github.com:danielgonzagat/penin-monorepo.git

# Configurar Git LFS
git lfs install
git lfs track "*.pt" "*.pth" "*.ckpt" "*.bin" "*.safetensors" "*.onnx"
git lfs track "*.h5" "*.hdf5" "*.pkl" "*.pickle" "*.npy" "*.npz"

# Commit inicial
git add .
git commit -m "feat: initial commit - PENIN system setup"
git push -u origin main
```

## Configuração do Sistema de Sincronização

### 1. Configurar Mapeamentos
Edite `/opt/penin-autosync/config/config.yaml` para incluir seus diretórios:

```yaml
mappings:
  - src: "/opt/et_ultimate"
    dst: "opt/et_ultimate"
    description: "Sistema ET Ultimate - Cérebro Principal"
    
  - src: "/root/projetos"
    dst: "projetos"
    description: "Projetos Diversos"
    
  - src: "/opt/ml"
    dst: "ml"
    description: "Machine Learning Models"
    
  - src: "/opt/penin_omega"
    dst: "penin_omega"
    description: "Sistema PENIN Omega"
    
  # Adicione mais mapeamentos conforme necessário
  - src: "/home/usuario/desenvolvimento"
    dst: "desenvolvimento"
    description: "Projetos de Desenvolvimento"
```

### 2. Configurar Exclusões
Ajuste os padrões de exclusão conforme suas necessidades:

```yaml
ignore_globs:
  # Adicione padrões específicos do seu ambiente
  - "**/meus_segredos/**"
  - "**/dados_sensíveis/**"
  - "**/backup_antigo/**"
```

## Configuração do Cursor API

### 1. Obter Chave API
1. Acesse [Cursor Dashboard](https://cursor.com/dashboard)
2. Vá para Integrations → API Keys
3. Crie uma nova chave API
4. Copie a chave (formato: `ghp_...`)

### 2. Configurar Agentes
Edite a seção `cursor_api` no `config.yaml`:

```yaml
cursor_api:
  enabled: true
  base_url: "https://api.cursor.com"
  
  agents:
    - name: "code-reviewer"
      prompt: "Review code changes and suggest improvements for better performance, security, and maintainability. Focus on Python best practices and clean code principles."
      trigger_on: ["pull_request", "push"]
      
    - name: "bug-fixer"
      prompt: "Automatically detect and fix bugs in the codebase. Focus on common Python issues like import errors, type mismatches, and logic errors."
      trigger_on: ["issue", "error_log"]
      
    - name: "documentation-updater"
      prompt: "Keep documentation updated with code changes. Ensure README files, docstrings, and comments are current and helpful."
      trigger_on: ["code_change", "new_feature"]
      
    - name: "security-scanner"
      prompt: "Scan for security vulnerabilities and suggest fixes. Focus on Python security best practices and common vulnerabilities."
      trigger_on: ["push", "pull_request"]
```

## Configuração do Systemd (Opcional)

### 1. Criar Serviço
```bash
sudo tee /etc/systemd/system/penin-sync.service > /dev/null << 'EOF'
[Unit]
Description=PENIN Auto Sync Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/penin-autosync
Environment=PATH=/opt/penin-autosync/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/opt/penin-autosync/venv/bin/python /opt/penin-autosync/tools/auto_sync.py /opt/penin-autosync/config/config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

### 2. Ativar Serviço
```bash
# Recarregar systemd
sudo systemctl daemon-reload

# Habilitar serviço
sudo systemctl enable penin-sync.service

# Iniciar serviço
sudo systemctl start penin-sync.service

# Verificar status
sudo systemctl status penin-sync.service

# Ver logs
sudo journalctl -u penin-sync.service -f
```

## Teste do Sistema

### 1. Teste Manual
```bash
# Ativar ambiente virtual
source /opt/penin-autosync/venv/bin/activate

# Executar sincronização manual
cd /opt/penin-autosync
python tools/auto_sync.py config/config.yaml
```

### 2. Teste de Mudanças
```bash
# Criar arquivo de teste
echo "# Teste do Sistema PENIN" > /opt/et_ultimate/teste.md

# Verificar se foi sincronizado
cd /opt/penin-monorepo
git status
git log --oneline -5
```

### 3. Verificar README
```bash
# Verificar se README foi gerado
cat /opt/penin-monorepo/README.md

# Verificar se foi commitado
git log --oneline -1
```

## Solução de Problemas

### Problemas Comuns

#### 1. Erro de Permissão SSH
```bash
# Verificar permissões
ls -la ~/.ssh/

# Corrigir permissões
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

#### 2. Erro de Conexão GitHub
```bash
# Testar conexão
ssh -T git@github.com

# Verificar configuração
git remote -v
```

#### 3. Erro de Dependências Python
```bash
# Recriar ambiente virtual
rm -rf /opt/penin-autosync/venv
python3 -m venv /opt/penin-autosync/venv
source /opt/penin-autosync/venv/bin/activate
pip install -r requirements.txt
```

#### 4. Erro de Sincronização
```bash
# Verificar logs
tail -f /opt/penin-autosync/logs/sync.log

# Verificar configuração
python -c "import yaml; print(yaml.safe_load(open('/opt/penin-autosync/config/config.yaml')))"
```

## Próximos Passos

Após a configuração inicial:

1. **Personalizar Mapeamentos**: Adicione seus diretórios específicos
2. **Configurar Agentes**: Ajuste os prompts dos agentes Cursor
3. **Testar Integração**: Verifique se tudo está funcionando
4. **Monitorar Logs**: Acompanhe o funcionamento do sistema
5. **Expandir Funcionalidades**: Adicione novos recursos conforme necessário

---

*Este guia é atualizado automaticamente conforme o sistema evolui.*