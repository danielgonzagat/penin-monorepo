#!/usr/bin/env python3
"""
Sistema de Sincronização Automática CPU → GitHub
PENIN Monorepo com README Mestre Auto-Atualizado

Este script monitora mudanças nos diretórios configurados e sincroniza
automaticamente com o repositório GitHub, mantendo o README sempre atualizado.
"""

import os
import sys
import yaml
import time
import json
import shutil
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import requests
from git import Repo, GitCommandError
import fnmatch

# Configuração de Logging
def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Configura o sistema de logging"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Criar diretório de logs se não existir
    log_file = log_config.get('file', '/opt/penin-autosync/logs/sync.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configurar logger
    logger = logging.getLogger('penin_sync')
    logger.setLevel(log_level)
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class FileChangeHandler(FileSystemEventHandler):
    """Handler para mudanças de arquivos"""
    
    def __init__(self, sync_manager):
        self.sync_manager = sync_manager
        self.logger = sync_manager.logger
        self.last_change = 0
        self.debounce_seconds = sync_manager.config['sync']['debounce_seconds']
        
    def on_modified(self, event):
        if not event.is_directory:
            self._handle_change(event.src_path)
            
    def on_created(self, event):
        if not event.is_directory:
            self._handle_change(event.src_path)
            
    def on_deleted(self, event):
        if not event.is_directory:
            self._handle_change(event.src_path)
            
    def on_moved(self, event):
        if not event.is_directory:
            self._handle_change(event.dest_path)
            
    def _handle_change(self, file_path: str):
        """Handle file change with debouncing"""
        current_time = time.time()
        
        # Verificar se o arquivo deve ser ignorado
        if self.sync_manager.should_ignore_file(file_path):
            return
            
        # Debounce
        if current_time - self.last_change < self.debounce_seconds:
            return
            
        self.last_change = current_time
        self.logger.info(f"Arquivo alterado: {file_path}")
        
        # Agendar sincronização
        self.sync_manager.schedule_sync()

class SyncManager:
    """Gerenciador principal de sincronização"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.logger = setup_logging(self.config)
        self.repo = None
        self.last_sync = 0
        self.coalesce_seconds = self.config['sync']['coalesce_seconds']
        self.sync_scheduled = False
        
        # Inicializar repositório
        self.init_repo()
        
        # Configurar handlers
        self.event_handler = FileChangeHandler(self)
        self.observer = Observer()
        
    def load_config(self) -> Dict[str, Any]:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            sys.exit(1)
            
    def init_repo(self):
        """Inicializa o repositório Git"""
        repo_path = self.config['repo']['path']
        
        try:
            if os.path.exists(os.path.join(repo_path, '.git')):
                self.repo = Repo(repo_path)
                self.logger.info(f"Repositório carregado: {repo_path}")
            else:
                self.logger.info(f"Inicializando novo repositório: {repo_path}")
                self.repo = Repo.init(repo_path)
                
                # Configurar remote
                remote_url = self.config['repo']['remote_url']
                try:
                    self.repo.create_remote('origin', remote_url)
                except Exception:
                    # Remote já existe
                    pass
                    
                # Configurar Git LFS
                self.setup_git_lfs()
                
        except Exception as e:
            self.logger.error(f"Erro ao inicializar repositório: {e}")
            sys.exit(1)
            
    def setup_git_lfs(self):
        """Configura Git LFS para arquivos grandes"""
        try:
            # Criar .gitattributes
            gitattributes_path = os.path.join(self.repo.working_dir, '.gitattributes')
            lfs_patterns = [
                "*.pt",
                "*.pth", 
                "*.ckpt",
                "*.bin",
                "*.safetensors",
                "*.onnx",
                "*.h5",
                "*.hdf5",
                "*.pb",
                "*.tflite",
                "*.mlmodel",
                "*.coreml",
                "*.pkl",
                "*.pickle",
                "*.joblib",
                "*.npy",
                "*.npz"
            ]
            
            with open(gitattributes_path, 'w') as f:
                for pattern in lfs_patterns:
                    f.write(f"{pattern} filter=lfs diff=lfs merge=lfs -text\n")
                    
            # Executar git lfs install
            subprocess.run(['git', 'lfs', 'install'], cwd=self.repo.working_dir, check=True)
            self.logger.info("Git LFS configurado")
            
        except Exception as e:
            self.logger.warning(f"Erro ao configurar Git LFS: {e}")
            
    def should_ignore_file(self, file_path: str) -> bool:
        """Verifica se o arquivo deve ser ignorado"""
        ignore_globs = self.config.get('ignore_globs', [])
        
        for pattern in ignore_globs:
            if fnmatch.fnmatch(file_path, pattern):
                return True
                
        return False
        
    def should_include_file(self, file_path: str) -> bool:
        """Verifica se o arquivo deve ser incluído"""
        include_globs = self.config.get('include_globs', [])
        
        for pattern in include_globs:
            if fnmatch.fnmatch(file_path, pattern):
                return True
                
        return False
        
    def sync_directories(self):
        """Sincroniza diretórios configurados"""
        mappings = self.config.get('mappings', [])
        
        for mapping in mappings:
            src = mapping['src']
            dst = mapping['dst']
            
            if not os.path.exists(src):
                self.logger.warning(f"Diretório fonte não existe: {src}")
                continue
                
            dst_path = os.path.join(self.repo.working_dir, dst)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            try:
                # Usar rsync para sincronização eficiente
                self.rsync_directory(src, dst_path)
                self.logger.info(f"Sincronizado: {src} → {dst}")
                
            except Exception as e:
                self.logger.error(f"Erro ao sincronizar {src}: {e}")
                
    def rsync_directory(self, src: str, dst: str):
        """Sincroniza diretório usando rsync"""
        # Construir lista de exclusões
        exclude_args = []
        for pattern in self.config.get('ignore_globs', []):
            exclude_args.extend(['--exclude', pattern])
            
        # Comando rsync
        cmd = [
            'rsync', '-av', '--delete',
            *exclude_args,
            f"{src}/",
            f"{dst}/"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erro no rsync: {e.stderr}")
            raise
            
    def build_readme(self):
        """Constrói o README mestre"""
        if not self.config['sync']['build_readme']:
            return
            
        try:
            # Importar o builder de README
            sys.path.append('/opt/penin-autosync/tools')
            from build_readme import build_readme
            
            build_readme(self.config, self.repo)
            self.logger.info("README construído com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao construir README: {e}")
            
    def commit_and_push(self):
        """Faz commit e push das mudanças"""
        try:
            # Adicionar todos os arquivos
            self.repo.git.add('.')
            
            # Verificar se há mudanças
            if not self.repo.index.diff("HEAD"):
                self.logger.info("Nenhuma mudança para commitar")
                return
                
            # Commit
            commit_msg = f"auto(sync): {datetime.now().isoformat()}"
            self.repo.index.commit(commit_msg)
            
            # Push
            origin = self.repo.remote('origin')
            origin.push()
            
            self.logger.info(f"Commit e push realizados: {commit_msg}")
            
            # Trigger Cursor API agents se configurado
            self.trigger_cursor_agents()
            
        except GitCommandError as e:
            self.logger.error(f"Erro no Git: {e}")
        except Exception as e:
            self.logger.error(f"Erro no commit/push: {e}")
            
    def trigger_cursor_agents(self):
        """Dispara agentes do Cursor API"""
        cursor_config = self.config.get('cursor_api', {})
        
        if not cursor_config.get('enabled', False):
            return
            
        api_key = os.getenv('CURSOR_API_KEY')
        if not api_key:
            self.logger.warning("CURSOR_API_KEY não configurada")
            return
            
        base_url = cursor_config.get('base_url', 'https://api.cursor.com')
        agents = cursor_config.get('agents', [])
        
        for agent in agents:
            try:
                self.create_cursor_agent(agent, base_url, api_key)
            except Exception as e:
                self.logger.error(f"Erro ao criar agente {agent['name']}: {e}")
                
    def create_cursor_agent(self, agent_config: Dict[str, Any], base_url: str, api_key: str):
        """Cria um agente do Cursor API"""
        url = f"{base_url}/agents"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'name': agent_config['name'],
            'prompt': agent_config['prompt'],
            'repository': f"{self.config['readme']['project']['github_user']}/{self.config['repo']['path'].split('/')[-1]}"
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        self.logger.info(f"Agente {agent_config['name']} criado com sucesso")
        
    def schedule_sync(self):
        """Agenda sincronização com coalescência"""
        if self.sync_scheduled:
            return
            
        self.sync_scheduled = True
        
        def delayed_sync():
            time.sleep(self.coalesce_seconds)
            self.perform_sync()
            self.sync_scheduled = False
            
        threading.Thread(target=delayed_sync, daemon=True).start()
        
    def perform_sync(self):
        """Executa sincronização completa"""
        self.logger.info("Iniciando sincronização...")
        
        try:
            # Sincronizar diretórios
            self.sync_directories()
            
            # Construir README
            self.build_readme()
            
            # Commit e push
            self.commit_and_push()
            
            self.last_sync = time.time()
            self.logger.info("Sincronização concluída")
            
        except Exception as e:
            self.logger.error(f"Erro na sincronização: {e}")
            
    def start_monitoring(self):
        """Inicia monitoramento de arquivos"""
        mappings = self.config.get('mappings', [])
        
        for mapping in mappings:
            src = mapping['src']
            if os.path.exists(src):
                self.observer.schedule(self.event_handler, src, recursive=True)
                self.logger.info(f"Monitorando: {src}")
                
        self.observer.start()
        
        # Sincronização inicial
        self.perform_sync()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Parando monitoramento...")
            self.observer.stop()
            self.observer.join()

def main():
    """Função principal"""
    if len(sys.argv) != 2:
        print("Uso: python auto_sync.py <config.yaml>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Arquivo de configuração não encontrado: {config_path}")
        sys.exit(1)
        
    # Criar gerenciador de sincronização
    sync_manager = SyncManager(config_path)
    
    # Iniciar monitoramento
    sync_manager.start_monitoring()

if __name__ == "__main__":
    main()