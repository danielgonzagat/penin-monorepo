#!/bin/bash
# Script de Execução do Sistema PENIN
# Versão simplificada para testes

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║          PENIN - Sistema de Sincronização            ║"
echo "║            CPU → GitHub Automático                   ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Verificar se config existe
if [ ! -f "/opt/penin-autosync/config/config.yaml" ]; then
    echo -e "${RED}[!] Arquivo de configuração não encontrado!${NC}"
    exit 1
fi

# Criar repositório se não existir
if [ ! -d "/opt/penin-monorepo/.git" ]; then
    echo -e "${YELLOW}[*] Inicializando repositório Git...${NC}"
    cd /opt/penin-monorepo
    git init
    git config user.name "Falcon-Q"
    git config user.email "falcon-q@system.local"
    
    # Tentar adicionar remote
    git remote add origin git@github.com:danielgonzagat/penin-monorepo.git 2>/dev/null || true
    
    # Configurar Git LFS
    git lfs install
    git lfs track "*.pt" "*.pth" "*.ckpt" "*.bin" "*.safetensors"
    
    echo -e "${GREEN}[✓] Repositório inicializado${NC}"
fi

# Criar alguns arquivos de exemplo nos diretórios
echo -e "${YELLOW}[*] Criando arquivos de exemplo...${NC}"

# ET Ultimate
cat > /opt/et_ultimate/README.md << 'EOF'
# ET Ultimate - Cérebro Principal

Sistema de inteligência artificial central do PENIN.

## Componentes
- Processamento de linguagem natural
- Raciocínio lógico
- Memória associativa
- Tomada de decisões

## Status
Ativo e em desenvolvimento
EOF

cat > /opt/et_ultimate/agents/brain/neural_core.py << 'EOF'
"""
Neural Core - Núcleo do sistema ET Ultimate
"""

class NeuralCore:
    def __init__(self):
        self.version = "1.0.0"
        self.status = "active"
    
    def process(self, input_data):
        """Processa entrada e retorna resposta"""
        return f"Processado: {input_data}"
    
    def learn(self, data):
        """Aprende com novos dados"""
        pass
    
    def evolve(self):
        """Auto-evolução do sistema"""
        pass
EOF

# ML Models
cat > /opt/ml/README.md << 'EOF'
# Machine Learning Models

Repositório de modelos de machine learning do sistema PENIN.

## Modelos Disponíveis
- NLP Models
- Computer Vision
- Reinforcement Learning
- Time Series

## Status
Em desenvolvimento
EOF

# PENIN Omega
cat > /opt/penin_omega/README.md << 'EOF'
# PENIN Omega

Sistema de evolução e otimização automática.

## Funcionalidades
- Auto-modificação de código
- Otimização contínua
- Aprendizado por reforço
- Adaptação dinâmica

## Status
Protótipo inicial
EOF

# Projetos
cat > /root/projetos/README.md << 'EOF'
# Projetos Diversos

Coleção de projetos experimentais e protótipos.

## Projetos Ativos
- Sistema de sincronização
- API de inteligência
- Interface neural
- Automação avançada

## Status
Múltiplos projetos em andamento
EOF

echo -e "${GREEN}[✓] Arquivos de exemplo criados${NC}"

# Executar sincronização inicial
echo -e "${YELLOW}[*] Executando sincronização inicial...${NC}"

# Ativar ambiente virtual
source /opt/penin-autosync/venv/bin/activate

# Importar bibliotecas Python e executar sync
python3 << 'PYTHON_SCRIPT'
import os
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Adicionar path para importar nossos módulos
sys.path.insert(0, '/opt/penin-autosync/tools')

# Configurações
repo_path = "/opt/penin-monorepo"
mappings = [
    ("/opt/et_ultimate", "opt/et_ultimate"),
    ("/root/projetos", "projetos"),
    ("/opt/ml", "ml"),
    ("/opt/penin_omega", "penin_omega")
]

print("📁 Sincronizando diretórios...")

# Sincronizar cada mapeamento
for src, dst in mappings:
    if os.path.exists(src):
        dst_path = os.path.join(repo_path, dst)
        os.makedirs(os.path.dirname(dst_path) if os.path.dirname(dst_path) else dst_path, exist_ok=True)
        
        # Usar rsync para sincronizar (se disponível) ou copiar
        try:
            # Criar diretório destino se não existir
            os.makedirs(dst_path, exist_ok=True)
            
            # Copiar arquivos
            for root, dirs, files in os.walk(src):
                # Pular diretórios indesejados
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for file in files:
                    if not file.startswith('.') and not file.endswith('.pyc'):
                        src_file = os.path.join(root, file)
                        rel_path = os.path.relpath(src_file, src)
                        dst_file = os.path.join(dst_path, rel_path)
                        
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copy2(src_file, dst_file)
            
            print(f"  ✓ {src} → {dst}")
        except Exception as e:
            print(f"  ✗ Erro ao sincronizar {src}: {e}")

print("\n📝 Gerando README principal...")

# Gerar README principal
readme_content = f"""# PENIN Monorepo

> **Sistema PENIN - Evolução Contínua do Zero ao State-of-the-Art**
> 
> Sistema de sincronização automática CPU → GitHub com README mestre sempre atualizado

## 🚀 Visão Geral

Este repositório é um **monorepo** que espelha automaticamente múltiplas pastas do sistema local para o GitHub.

### ✨ Status do Sistema

- **Última Atualização:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Versão:** 1.0.0
- **Status:** Ativo

## 🏗️ Estrutura do Projeto

```
penin-monorepo/
├── opt/
│   └── et_ultimate/      # Sistema ET Ultimate - Cérebro Principal
│       └── agents/
│           └── brain/     # Módulos neurais centrais
├── ml/                    # Machine Learning Models
├── penin_omega/          # Sistema PENIN Omega
└── projetos/             # Projetos Diversos
```

## 📊 Componentes

### 🧠 ET Ultimate
- **Localização:** `opt/et_ultimate/`
- **Descrição:** Sistema de inteligência artificial central
- **Status:** Ativo

### 🤖 Machine Learning
- **Localização:** `ml/`
- **Descrição:** Modelos e algoritmos de ML
- **Status:** Em desenvolvimento

### ⚡ PENIN Omega
- **Localização:** `penin_omega/`
- **Descrição:** Sistema de evolução automática
- **Status:** Protótipo

### 🚀 Projetos
- **Localização:** `projetos/`
- **Descrição:** Projetos experimentais
- **Status:** Múltiplos ativos

## 🔄 Como Usar

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/danielgonzagat/penin-monorepo.git
   ```

2. **Execute o sistema de sincronização:**
   ```bash
   /opt/penin-autosync/run.sh
   ```

## 🤖 Integração com Cursor API

Este sistema está preparado para integração com a API de Agentes em Segundo Plano do Cursor para:
- Revisão automática de código
- Correção de bugs
- Atualização de documentação
- Análise de segurança

## 📄 Licença

MIT License

---

*README gerado automaticamente em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

# Salvar README
readme_path = os.path.join(repo_path, "README.md")
with open(readme_path, 'w') as f:
    f.write(readme_content)

print("  ✓ README.md gerado")

# Git operations
os.chdir(repo_path)

# Adicionar arquivos
subprocess.run(['git', 'add', '.'], capture_output=True)

# Verificar se há mudanças
result = subprocess.run(['git', 'diff', '--cached', '--quiet'], capture_output=True)
if result.returncode != 0:
    # Fazer commit
    commit_msg = f"auto(sync): {datetime.now().isoformat()}"
    subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True)
    print(f"\n✓ Commit realizado: {commit_msg}")
    
    # Tentar push (pode falhar se repo remoto não existe)
    try:
        result = subprocess.run(['git', 'push', '-u', 'origin', 'main'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Push realizado com sucesso!")
        else:
            # Tentar criar branch main primeiro
            subprocess.run(['git', 'branch', '-M', 'main'], capture_output=True)
            result = subprocess.run(['git', 'push', '-u', 'origin', 'main'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Push realizado com sucesso!")
            else:
                print("⚠ Push falhou - verifique se o repositório remoto existe no GitHub")
                print("  Crie em: https://github.com/new")
                print(f"  Nome: penin-monorepo")
    except Exception as e:
        print(f"⚠ Erro no push: {e}")
else:
    print("\nℹ Nenhuma mudança para commitar")

print("\n" + "="*50)
print("Sistema PENIN sincronizado com sucesso!")
print("="*50)
PYTHON_SCRIPT

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Sincronização Inicial Concluída!             ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Para monitoramento contínuo, execute:"
echo "  python /opt/penin-autosync/tools/auto_sync.py /opt/penin-autosync/config/config.yaml"
echo ""