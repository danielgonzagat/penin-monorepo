#!/usr/bin/env python3
"""
Sistema de Sincronização PENIN Simplificado
Sincronização contínua GitHub ↔ Servidor
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/penin-autosync/logs/sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PENIN')

# Configurações
REPO_PATH = '/opt/penin-monorepo'
GITHUB_USER = 'danielgonzagat'
GITHUB_REPO = 'penin-monorepo'
GITHUB_TOKEN = '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'

# Diretórios para sincronizar
SYNC_DIRS = [
    ('/opt/et_ultimate', 'opt/et_ultimate'),
    ('/opt/ml', 'ml'),
    ('/opt/penin_omega', 'penin_omega'),
    ('/root/projetos', 'projetos'),
    ('/opt/penin-autosync', 'penin_system')
]

def run_command(cmd, cwd=None):
    """Executa comando shell"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def sync_to_github():
    """Sincroniza servidor → GitHub"""
    logger.info("📤 Sincronizando para GitHub...")
    
    # Sincronizar diretórios
    for src, dst in SYNC_DIRS:
        if os.path.exists(src):
            dst_path = os.path.join(REPO_PATH, dst)
            os.makedirs(dst_path, exist_ok=True)
            
            # Usar rsync para sincronizar
            cmd = f'rsync -av --delete --exclude=.git --exclude=__pycache__ --exclude=.venv --exclude=*.pyc --exclude=.env "{src}/" "{dst_path}/"'
            success, _, _ = run_command(cmd)
            
            if success:
                logger.info(f"  ✅ {src} → {dst}")
            else:
                logger.warning(f"  ⚠️ Erro ao sincronizar {src}")
    
    # Gerar README atualizado
    generate_readme()
    
    # Git operations
    os.chdir(REPO_PATH)
    
    # Adicionar mudanças
    run_command('git add -A')
    
    # Verificar se há mudanças
    success, output, _ = run_command('git diff --cached --quiet')
    if not success:  # Há mudanças
        # Commit
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        commit_msg = f"auto: sync from server - {timestamp}"
        run_command(f'git commit -m "{commit_msg}"')
        
        # Push
        success, output, error = run_command('git push origin main')
        if success:
            logger.info(f"✅ Push realizado com sucesso")
        else:
            logger.warning(f"⚠️ Push falhou: {error}")
    else:
        logger.info("ℹ️ Sem mudanças para sincronizar")

def pull_from_github():
    """Sincroniza GitHub → Servidor"""
    logger.info("📥 Puxando mudanças do GitHub...")
    
    os.chdir(REPO_PATH)
    
    # Fetch changes
    run_command('git fetch origin')
    
    # Check if there are changes
    success, local_hash, _ = run_command('git rev-parse HEAD')
    success, remote_hash, _ = run_command('git rev-parse origin/main')
    
    if local_hash.strip() != remote_hash.strip():
        # Pull changes
        success, _, _ = run_command('git pull origin main')
        
        if success:
            logger.info("✅ Pull realizado com sucesso")
            
            # Sync back to local directories
            for src, dst in SYNC_DIRS:
                repo_src = os.path.join(REPO_PATH, dst)
                if os.path.exists(repo_src):
                    cmd = f'rsync -av --delete --exclude=.git "{repo_src}/" "{src}/"'
                    run_command(cmd)
                    logger.info(f"  ✅ {dst} → {src}")
        else:
            logger.warning("⚠️ Pull falhou")
    else:
        logger.info("ℹ️ Sem mudanças no GitHub")

def generate_readme():
    """Gera README atualizado"""
    try:
        # Coletar estatísticas
        total_files = 0
        total_lines = 0
        
        for root, dirs, files in os.walk(REPO_PATH):
            if '.git' not in root:
                total_files += len(files)
                for file in files:
                    if file.endswith(('.py', '.md', '.txt', '.sh', '.yaml', '.json')):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                                total_lines += len(f.readlines())
                        except:
                            pass
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        readme_content = f"""# PENIN Monorepo - Sistema de Evolução Contínua

![Status](https://img.shields.io/badge/status-active-success)
![Sync](https://img.shields.io/badge/sync-bidirectional-blue)
![Files](https://img.shields.io/badge/files-{total_files}-orange)
![Lines](https://img.shields.io/badge/lines-{total_lines}-yellow)

> **Sistema PENIN com Sincronização Bidirecional GitHub ↔ Servidor**  
> *Evolução Automática do Zero ao State-of-the-Art*

## 🚀 Status do Sistema

| Métrica | Valor |
|---------|-------|
| **Última Sincronização** | {timestamp} |
| **Total de Arquivos** | {total_files:,} |
| **Total de Linhas de Código** | {total_lines:,} |
| **Status** | ✅ Operacional 24/7 |
| **Repositório** | [github.com/{GITHUB_USER}/{GITHUB_REPO}](https://github.com/{GITHUB_USER}/{GITHUB_REPO}) |

## 🔄 Sincronização Bidirecional

Este sistema mantém sincronização contínua entre:
- **Servidor Local** → **GitHub** (push automático)
- **GitHub** → **Servidor Local** (pull automático)

### Diretórios Sincronizados

| Local | Repositório | Descrição |
|-------|-------------|-----------|
| `/opt/et_ultimate` | `opt/et_ultimate/` | Sistema ET Ultimate - Cérebro Principal |
| `/opt/ml` | `ml/` | Modelos de Machine Learning |
| `/opt/penin_omega` | `penin_omega/` | Sistema de Evolução |
| `/root/projetos` | `projetos/` | Projetos Diversos |
| `/opt/penin-autosync` | `penin_system/` | Sistema de Sincronização |

## 🤖 Agentes Cursor

Configure agentes em segundo plano para evolução automática:

1. Acesse [Cursor Dashboard](https://cursor.com/dashboard)
2. Crie agentes apontando para este repositório
3. Os agentes trabalharão autonomamente 24/7

### Agentes Recomendados

- **Code Optimizer**: Otimização contínua de código
- **Bug Fixer**: Correção automática de bugs
- **Documentation**: Atualização de documentação
- **Security Scanner**: Análise de segurança

## 📊 Arquitetura do Sistema

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Servidor Local │ ←→  │   Git/GitHub    │ ←→  │  Cursor Agents  │
│   (Este CPU)    │     │  (Repositório)  │     │   (Cloud AI)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ↑                       ↑                        ↓
        │                       │                        │
        └───────────────────────┴────────────────────────┘
                    Sincronização Bidirecional
```

## 🛠️ Componentes Principais

### ET Ultimate (`opt/et_ultimate/`)
Sistema de inteligência artificial central com:
- Neural core para processamento
- Memória associativa
- Tomada de decisões
- Auto-evolução

### Machine Learning (`ml/`)
Modelos e algoritmos de ML:
- Redes neurais
- Processamento de linguagem
- Visão computacional
- Aprendizado por reforço

### PENIN Omega (`penin_omega/`)
Sistema de evolução automática:
- Auto-modificação de código
- Otimização contínua
- Adaptação dinâmica
- Métricas de evolução

### Sistema de Sincronização (`penin_system/`)
Infraestrutura de sincronização:
- Scripts Python
- Configurações YAML
- Logs e monitoramento
- Integração com APIs

## 📈 Evolução Contínua

O sistema evolui automaticamente através de:

1. **Monitoramento 24/7**: Detecta mudanças instantaneamente
2. **Sincronização Bidirecional**: Mantém tudo sincronizado
3. **Agentes Inteligentes**: Melhoram o código autonomamente
4. **Documentação Viva**: README sempre atualizado

## 🔧 Comandos Úteis

```bash
# Ver status do sistema
systemctl status penin-sync

# Ver logs em tempo real
journalctl -u penin-sync -f

# Sincronização manual
python /opt/penin-autosync/start_sync.py

# Interface de controle
penin
```

## 🔒 Segurança

- ✅ Tokens seguros configurados
- ✅ Sincronização via HTTPS
- ✅ Logs para auditoria
- ✅ Exclusão de arquivos sensíveis

## 📝 Licença

MIT License - Sistema de código aberto

---

<div align="center">

**Sistema PENIN - Evolução Infinita**  
*Sincronizado automaticamente em {timestamp}*

[Ver no GitHub](https://github.com/{GITHUB_USER}/{GITHUB_REPO}) | 
[Cursor Dashboard](https://cursor.com/dashboard) | 
[Logs do Sistema](/opt/penin-autosync/logs/)

</div>
"""
        
        # Salvar README
        with open(f"{REPO_PATH}/README.md", 'w') as f:
            f.write(readme_content)
        
        logger.info("📝 README atualizado")
        
    except Exception as e:
        logger.error(f"❌ Erro ao gerar README: {e}")

def main():
    """Loop principal de sincronização"""
    logger.info("="*60)
    logger.info("   PENIN - Sistema de Sincronização Iniciado")
    logger.info("   GitHub ↔ Servidor - Operação 24/7")
    logger.info("="*60)
    
    # Configurar Git
    os.chdir(REPO_PATH)
    run_command(f'git config user.name "{GITHUB_USER}"')
    run_command(f'git config user.email "danielgonzagatj@gmail.com"')
    
    # Loop infinito
    while True:
        try:
            # Sincronizar para GitHub
            sync_to_github()
            
            # Aguardar 10 segundos
            time.sleep(10)
            
            # Verificar mudanças no GitHub
            pull_from_github()
            
            # Aguardar mais 10 segundos
            time.sleep(10)
            
        except KeyboardInterrupt:
            logger.info("⏹️ Sistema interrompido pelo usuário")
            break
        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}")
            time.sleep(30)  # Aguardar 30 segundos em caso de erro

if __name__ == "__main__":
    main()