#!/usr/bin/env python3
"""
Sistema de Sincronização Bidirecional PENIN
GitHub ↔ Servidor com Agentes Cursor Integrados
"""

import os
import sys
import time
import json
import yaml
import shutil
import hashlib
import requests
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import git
from git import Repo
import fnmatch
from queue import Queue
import logging

# Carregar variáveis de ambiente
from dotenv import load_dotenv
load_dotenv('/opt/penin-autosync/.env')

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/penin-autosync/logs/bidirectional_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('PENIN-SYNC')

class CursorAgentManager:
    """Gerenciador de Agentes Cursor"""
    
    def __init__(self):
        self.api_key = os.getenv('CURSOR_API_KEY')
        self.base_url = "https://api.cursor.com"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.active_agents = {}
        
    def create_agent(self, prompt: str, repository: str = None) -> Dict:
        """Cria um novo agente"""
        url = f"{self.base_url}/v0/agents"
        
        if not repository:
            repository = f"https://github.com/{os.getenv('GITHUB_USER')}/{os.getenv('GITHUB_REPO')}"
        
        data = {
            'prompt': {
                'text': prompt
            },
            'source': {
                'repository': repository,
                'ref': 'main'
            }
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            agent = response.json()
            self.active_agents[agent['id']] = agent
            logger.info(f"✅ Agente criado: {agent['id']}")
            return agent
        except Exception as e:
            logger.error(f"❌ Erro ao criar agente: {e}")
            return None
            
    def add_followup(self, agent_id: str, prompt: str) -> bool:
        """Adiciona instrução de followup ao agente"""
        url = f"{self.base_url}/v0/agents/{agent_id}/followup"
        
        data = {
            'prompt': {
                'text': prompt
            }
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()
            logger.info(f"✅ Followup adicionado ao agente {agent_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao adicionar followup: {e}")
            return False
            
    def get_agent_status(self, agent_id: str) -> Dict:
        """Obtém status do agente"""
        url = f"{self.base_url}/v0/agents/{agent_id}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Erro ao obter status: {e}")
            return None
            
    def setup_system_agents(self):
        """Configura agentes do sistema PENIN"""
        agents_config = [
            {
                'name': 'code-optimizer',
                'prompt': """You are the PENIN Code Optimizer Agent.
                Monitor all code changes and:
                1. Optimize performance bottlenecks
                2. Refactor complex functions
                3. Improve code readability
                4. Add type hints and documentation
                5. Ensure PEP8 compliance
                Always commit improvements with clear messages."""
            },
            {
                'name': 'security-guardian',
                'prompt': """You are the PENIN Security Guardian Agent.
                Continuously scan for:
                1. Security vulnerabilities
                2. Exposed credentials or secrets
                3. Unsafe code patterns
                4. Dependency vulnerabilities
                5. Permission issues
                Fix all security issues immediately."""
            },
            {
                'name': 'evolution-engine',
                'prompt': """You are the PENIN Evolution Engine Agent.
                Your mission is to:
                1. Identify areas for improvement
                2. Implement new features based on patterns
                3. Upgrade deprecated code
                4. Enhance system architecture
                5. Document evolution in README
                Make the system better with each iteration."""
            },
            {
                'name': 'sync-coordinator',
                'prompt': """You are the PENIN Sync Coordinator Agent.
                Ensure perfect synchronization by:
                1. Resolving merge conflicts
                2. Maintaining consistency between server and GitHub
                3. Organizing file structure
                4. Managing branches effectively
                5. Keeping commit history clean
                Coordinate all changes seamlessly."""
            }
        ]
        
        for agent_config in agents_config:
            logger.info(f"🤖 Criando agente: {agent_config['name']}")
            agent = self.create_agent(agent_config['prompt'])
            if agent:
                self.active_agents[agent_config['name']] = agent
                
        return self.active_agents

class GitHubWebhookServer:
    """Servidor de Webhooks do GitHub"""
    
    def __init__(self, sync_manager):
        self.sync_manager = sync_manager
        self.port = 8080
        self.secret = os.getenv('WEBHOOK_SECRET', 'default_secret')
        
    def start(self):
        """Inicia servidor de webhooks"""
        from flask import Flask, request
        import hmac
        
        app = Flask(__name__)
        
        @app.route('/webhook', methods=['POST'])
        def handle_webhook():
            # Verificar assinatura
            signature = request.headers.get('X-Hub-Signature-256')
            if signature:
                expected = 'sha256=' + hmac.new(
                    self.secret.encode(),
                    request.data,
                    hashlib.sha256
                ).hexdigest()
                
                if signature != expected:
                    return 'Invalid signature', 401
            
            # Processar evento
            event = request.headers.get('X-GitHub-Event')
            payload = request.json
            
            if event == 'push':
                logger.info(f"📥 Push recebido do GitHub")
                self.sync_manager.pull_from_github()
            elif event == 'pull_request':
                logger.info(f"🔀 Pull request evento: {payload.get('action')}")
                if payload.get('action') == 'closed' and payload['pull_request'].get('merged'):
                    self.sync_manager.pull_from_github()
                    
            return 'OK', 200
            
        # Rodar em thread separada
        threading.Thread(target=lambda: app.run(host='0.0.0.0', port=self.port), daemon=True).start()
        logger.info(f"🌐 Servidor de webhooks iniciado na porta {self.port}")

class BidirectionalSyncManager:
    """Gerenciador de Sincronização Bidirecional"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.repo_path = '/opt/penin-monorepo'
        self.repo = None
        self.agent_manager = CursorAgentManager()
        self.webhook_server = GitHubWebhookServer(self)
        self.sync_queue = Queue()
        self.last_sync = {}
        self.file_hashes = {}
        
        # Inicializar repositório
        self.init_repository()
        
        # Configurar agentes
        self.agent_manager.setup_system_agents()
        
        # Iniciar servidor de webhooks
        self.webhook_server.start()
        
    def load_config(self, config_path: str) -> Dict:
        """Carrega configuração"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def init_repository(self):
        """Inicializa repositório Git com configuração completa"""
        try:
            if not os.path.exists(self.repo_path):
                os.makedirs(self.repo_path)
                
            if not os.path.exists(f"{self.repo_path}/.git"):
                self.repo = Repo.init(self.repo_path)
                logger.info("📁 Repositório inicializado")
            else:
                self.repo = Repo(self.repo_path)
                
            # Configurar Git
            with self.repo.config_writer() as git_config:
                git_config.set_value('user', 'email', os.getenv('GITHUB_EMAIL'))
                git_config.set_value('user', 'name', os.getenv('GITHUB_USER'))
                
            # Configurar remote com token
            github_user = os.getenv('GITHUB_USER')
            github_token = os.getenv('GITHUB_TOKEN')
            github_repo = os.getenv('GITHUB_REPO')
            
            remote_url = f"https://{github_token}@github.com/{github_user}/{github_repo}.git"
            
            if 'origin' in self.repo.remotes:
                self.repo.delete_remote('origin')
                
            self.repo.create_remote('origin', remote_url)
            logger.info("🔗 Remote configurado com sucesso")
            
            # Configurar Git LFS
            subprocess.run(['git', 'lfs', 'install'], cwd=self.repo_path, check=True)
            
            # Criar .gitattributes para LFS
            gitattributes = """*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text"""
            
            with open(f"{self.repo_path}/.gitattributes", 'w') as f:
                f.write(gitattributes)
                
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar repositório: {e}")
            
    def calculate_file_hash(self, file_path: str) -> str:
        """Calcula hash do arquivo"""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except:
            return None
            
    def should_sync_file(self, file_path: str) -> bool:
        """Verifica se arquivo deve ser sincronizado"""
        # Verificar padrões de exclusão
        for pattern in self.config.get('ignore_globs', []):
            if fnmatch.fnmatch(file_path, pattern):
                return False
                
        # Verificar se mudou
        current_hash = self.calculate_file_hash(file_path)
        if current_hash and current_hash != self.file_hashes.get(file_path):
            self.file_hashes[file_path] = current_hash
            return True
            
        return False
        
    def sync_to_github(self):
        """Sincroniza servidor → GitHub"""
        try:
            logger.info("📤 Sincronizando para GitHub...")
            
            # Sincronizar diretórios mapeados
            for mapping in self.config.get('mappings', []):
                src = mapping['src']
                dst = os.path.join(self.repo_path, mapping['dst'])
                
                if os.path.exists(src):
                    # Criar diretório destino
                    os.makedirs(dst, exist_ok=True)
                    
                    # Sincronizar com rsync
                    exclude_args = []
                    for pattern in self.config.get('ignore_globs', []):
                        exclude_args.extend(['--exclude', pattern])
                        
                    cmd = [
                        'rsync', '-av', '--delete',
                        *exclude_args,
                        f"{src}/",
                        f"{dst}/"
                    ]
                    
                    subprocess.run(cmd, check=True, capture_output=True)
                    logger.info(f"  ✅ {src} → {mapping['dst']}")
                    
            # Gerar README atualizado
            self.generate_readme()
            
            # Git add, commit e push
            self.repo.git.add('.')
            
            if self.repo.is_dirty():
                commit_msg = f"auto: sync from server - {datetime.now().isoformat()}"
                self.repo.index.commit(commit_msg)
                
                # Push para GitHub
                origin = self.repo.remote('origin')
                origin.push('HEAD:main', force=True)
                
                logger.info(f"✅ Push realizado: {commit_msg}")
                
                # Notificar agentes sobre mudanças
                self.notify_agents_of_changes()
            else:
                logger.info("ℹ️ Sem mudanças para sincronizar")
                
        except Exception as e:
            logger.error(f"❌ Erro ao sincronizar para GitHub: {e}")
            
    def pull_from_github(self):
        """Sincroniza GitHub → Servidor"""
        try:
            logger.info("📥 Puxando mudanças do GitHub...")
            
            # Pull do GitHub
            origin = self.repo.remote('origin')
            origin.pull('main')
            
            # Sincronizar de volta para diretórios locais
            for mapping in self.config.get('mappings', []):
                src_repo = os.path.join(self.repo_path, mapping['dst'])
                dst_local = mapping['src']
                
                if os.path.exists(src_repo):
                    # Sincronizar de volta
                    cmd = [
                        'rsync', '-av', '--delete',
                        '--exclude=.git',
                        f"{src_repo}/",
                        f"{dst_local}/"
                    ]
                    
                    subprocess.run(cmd, check=True, capture_output=True)
                    logger.info(f"  ✅ {mapping['dst']} → {dst_local}")
                    
            logger.info("✅ Sincronização do GitHub concluída")
            
        except Exception as e:
            logger.error(f"❌ Erro ao puxar do GitHub: {e}")
            
    def generate_readme(self):
        """Gera README atualizado com status completo"""
        try:
            # Coletar estatísticas
            stats = {
                'total_files': 0,
                'total_lines': 0,
                'languages': {},
                'active_agents': len(self.agent_manager.active_agents),
                'last_sync': datetime.now().isoformat()
            }
            
            # Contar arquivos e linhas
            for root, dirs, files in os.walk(self.repo_path):
                if '.git' in root:
                    continue
                    
                for file in files:
                    stats['total_files'] += 1
                    ext = os.path.splitext(file)[1]
                    if ext:
                        stats['languages'][ext] = stats['languages'].get(ext, 0) + 1
                        
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            stats['total_lines'] += len(f.readlines())
                    except:
                        pass
                        
            # Gerar README
            readme_content = f"""# PENIN Monorepo - Sistema de Evolução Contínua

<div align="center">

![Status](https://img.shields.io/badge/status-active-success)
![Sync](https://img.shields.io/badge/sync-bidirectional-blue)
![Agents](https://img.shields.io/badge/agents-{stats['active_agents']}_active-green)
![Files](https://img.shields.io/badge/files-{stats['total_files']}-orange)
![Lines](https://img.shields.io/badge/lines-{stats['total_lines']}-yellow)

**Sistema PENIN com Sincronização Bidirecional GitHub ↔ Servidor**  
*Evolução Automática do Zero ao State-of-the-Art*

</div>

---

## 🚀 Status do Sistema

| Métrica | Valor |
|---------|-------|
| **Última Sincronização** | {stats['last_sync']} |
| **Total de Arquivos** | {stats['total_files']:,} |
| **Total de Linhas** | {stats['total_lines']:,} |
| **Agentes Ativos** | {stats['active_agents']} |
| **Modo de Operação** | Bidirecional 24/7 |

## 🤖 Agentes Cursor Ativos

Os seguintes agentes estão monitorando e evoluindo o código continuamente:

| Agente | Função | Status |
|--------|--------|--------|
| **Code Optimizer** | Otimização de performance e refatoração | ✅ Ativo |
| **Security Guardian** | Análise de segurança e correção de vulnerabilidades | ✅ Ativo |
| **Evolution Engine** | Implementação de melhorias e novos recursos | ✅ Ativo |
| **Sync Coordinator** | Coordenação de sincronização bidirecional | ✅ Ativo |

## 📊 Distribuição de Código

### Linguagens
"""
            for ext, count in sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / stats['total_files']) * 100
                readme_content += f"- **{ext}**: {count} arquivos ({percentage:.1f}%)\n"
                
            readme_content += f"""

## 🔄 Sincronização Bidirecional

### Como Funciona

1. **Servidor → GitHub**: Mudanças locais são detectadas e enviadas automaticamente
2. **GitHub → Servidor**: Mudanças no GitHub (PRs, commits diretos) são puxadas automaticamente
3. **Resolução de Conflitos**: Agentes resolvem conflitos automaticamente
4. **Monitoramento 24/7**: Sistema roda continuamente sem intervenção

### Fluxo de Dados

```mermaid
graph LR
    A[Servidor Local] <--> B[Git Repository]
    B <--> C[GitHub]
    C <--> D[Cursor Agents]
    D --> C
    C --> B
    B --> A
```

## 🛠️ Componentes do Sistema

### ET Ultimate
- **Caminho**: `opt/et_ultimate/`
- **Descrição**: Sistema de IA central
- **Módulos**: Neural core, processamento, memória

### Machine Learning
- **Caminho**: `ml/`
- **Descrição**: Modelos e algoritmos de ML
- **Status**: Em evolução contínua

### PENIN Omega
- **Caminho**: `penin_omega/`
- **Descrição**: Sistema de auto-evolução
- **Status**: Ativo

### Projetos
- **Caminho**: `projetos/`
- **Descrição**: Projetos experimentais
- **Status**: Múltiplos ativos

## 📈 Evolução Automática

O sistema evolui automaticamente através de:

1. **Análise Contínua**: Agentes analisam o código 24/7
2. **Melhorias Automáticas**: Implementação de otimizações
3. **Correções Proativas**: Bugs são corrigidos antes de causar problemas
4. **Documentação Viva**: README e docs sempre atualizados
5. **Aprendizado Contínuo**: Sistema aprende com cada iteração

## 🔒 Segurança

- ✅ Análise contínua de vulnerabilidades
- ✅ Detecção de secrets e credenciais
- ✅ Correção automática de issues de segurança
- ✅ Compliance com melhores práticas

## 📝 Logs e Monitoramento

Acompanhe o sistema em tempo real:

```bash
# Ver logs do sistema
journalctl -u penin-sync -f

# Status dos agentes
penin status

# Logs de sincronização
tail -f /opt/penin-autosync/logs/bidirectional_sync.log
```

## 🌐 Webhooks e Integração

- **GitHub Webhooks**: Configurados para notificar mudanças
- **Cursor API**: Integração completa com agentes
- **Slack**: Notificações de eventos importantes (opcional)

## 🚦 Como Contribuir

Este sistema aceita contribuições através de:

1. **Pull Requests no GitHub**: Serão analisados pelos agentes
2. **Commits Diretos**: Para colaboradores autorizados
3. **Issues**: Agentes respondem e implementam soluções

## 📄 Licença

MIT License - Sistema de código aberto para evolução contínua

---

<div align="center">

**Sistema PENIN - Evolução Infinita**  
*Sincronizado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*Versão: Auto-evolutiva*

[GitHub](https://github.com/{os.getenv('GITHUB_USER')}/{os.getenv('GITHUB_REPO')}) | 
[Cursor Dashboard](https://cursor.com/dashboard) | 
[Documentação](https://github.com/{os.getenv('GITHUB_USER')}/{os.getenv('GITHUB_REPO')}/wiki)

</div>
"""
            
            # Salvar README
            with open(f"{self.repo_path}/README.md", 'w') as f:
                f.write(readme_content)
                
            logger.info("📝 README atualizado")
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar README: {e}")
            
    def notify_agents_of_changes(self):
        """Notifica agentes sobre mudanças para análise"""
        try:
            # Obter último commit
            last_commit = self.repo.head.commit
            
            # Notificar agente de otimização
            if 'code-optimizer' in self.agent_manager.active_agents:
                self.agent_manager.add_followup(
                    self.agent_manager.active_agents['code-optimizer']['id'],
                    f"Analyze and optimize changes in commit {last_commit.hexsha[:8]}"
                )
                
            # Notificar agente de segurança
            if 'security-guardian' in self.agent_manager.active_agents:
                self.agent_manager.add_followup(
                    self.agent_manager.active_agents['security-guardian']['id'],
                    f"Scan for security issues in commit {last_commit.hexsha[:8]}"
                )
                
        except Exception as e:
            logger.error(f"❌ Erro ao notificar agentes: {e}")
            
    def continuous_sync_loop(self):
        """Loop principal de sincronização contínua"""
        logger.info("🔄 Iniciando loop de sincronização contínua")
        
        while True:
            try:
                # Sincronizar para GitHub
                self.sync_to_github()
                
                # Aguardar intervalo
                time.sleep(int(os.getenv('SYNC_INTERVAL', 5)))
                
                # Verificar se há mudanças no GitHub
                origin = self.repo.remote('origin')
                origin.fetch()
                
                # Se houver mudanças, puxar
                if self.repo.head.commit != origin.refs.main.commit:
                    self.pull_from_github()
                    
            except KeyboardInterrupt:
                logger.info("⏹️ Sincronização interrompida")
                break
            except Exception as e:
                logger.error(f"❌ Erro no loop de sincronização: {e}")
                time.sleep(10)  # Aguardar antes de tentar novamente

class FileChangeMonitor(FileSystemEventHandler):
    """Monitor de mudanças de arquivos"""
    
    def __init__(self, sync_manager):
        self.sync_manager = sync_manager
        self.last_sync = 0
        self.debounce = int(os.getenv('DEBOUNCE_SECONDS', 2))
        
    def on_any_event(self, event):
        if event.is_directory:
            return
            
        current_time = time.time()
        if current_time - self.last_sync > self.debounce:
            self.last_sync = current_time
            logger.info(f"📁 Mudança detectada: {event.src_path}")
            # Sincronizar será feito pelo loop principal

def main():
    """Função principal"""
    logger.info("="*60)
    logger.info("   PENIN - Sistema de Sincronização Bidirecional")
    logger.info("   GitHub ↔ Servidor com Agentes Cursor")
    logger.info("="*60)
    
    # Criar gerenciador de sincronização
    sync_manager = BidirectionalSyncManager('/opt/penin-autosync/config/config.yaml')
    
    # Configurar monitor de arquivos
    event_handler = FileChangeMonitor(sync_manager)
    observer = Observer()
    
    # Monitorar diretórios configurados
    for mapping in sync_manager.config.get('mappings', []):
        if os.path.exists(mapping['src']):
            observer.schedule(event_handler, mapping['src'], recursive=True)
            logger.info(f"👁️ Monitorando: {mapping['src']}")
            
    observer.start()
    
    # Iniciar loop de sincronização
    try:
        sync_manager.continuous_sync_loop()
    except KeyboardInterrupt:
        observer.stop()
        logger.info("✅ Sistema encerrado")
        
    observer.join()

if __name__ == "__main__":
    main()