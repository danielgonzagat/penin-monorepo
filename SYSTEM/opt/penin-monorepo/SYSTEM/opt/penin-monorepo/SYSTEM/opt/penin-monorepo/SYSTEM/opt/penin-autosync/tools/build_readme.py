#!/usr/bin/env python3
"""
Construtor de README Mestre
Gera automaticamente o README principal do monorepo PENIN
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from jinja2 import Environment, FileSystemLoader
import yaml

def get_git_info(repo_path: str) -> Dict[str, Any]:
    """Obtém informações do Git"""
    try:
        # Último commit
        result = subprocess.run(
            ['git', 'log', '-1', '--pretty=format:%H|%an|%ae|%ad|%s'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            parts = result.stdout.split('|')
            return {
                'last_commit': {
                    'hash': parts[0][:8],
                    'author': parts[1],
                    'email': parts[2],
                    'date': parts[3],
                    'message': parts[4]
                }
            }
    except subprocess.CalledProcessError:
        pass
        
    return {'last_commit': None}

def get_repo_stats(repo_path: str) -> Dict[str, Any]:
    """Obtém estatísticas do repositório"""
    stats = {
        'total_files': 0,
        'total_lines': 0,
        'languages': {},
        'directories': {},
        'recent_commits': []
    }
    
    try:
        # Contar arquivos por tipo
        for root, dirs, files in os.walk(repo_path):
            # Pular .git
            if '.git' in root:
                continue
                
            rel_path = os.path.relpath(root, repo_path)
            if rel_path == '.':
                continue
                
            # Contar arquivos por diretório
            stats['directories'][rel_path] = len(files)
            
            # Contar linhas e tipos
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Contar linhas
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        stats['total_lines'] += lines
                        
                    # Detectar linguagem
                    ext = os.path.splitext(file)[1].lower()
                    if ext:
                        stats['languages'][ext] = stats['languages'].get(ext, 0) + 1
                        
                    stats['total_files'] += 1
                    
                except Exception:
                    continue
                    
        # Obter commits recentes
        result = subprocess.run(
            ['git', 'log', '-10', '--pretty=format:%ad|%s|%an', '--date=short'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|', 2)
                    if len(parts) >= 3:
                        stats['recent_commits'].append({
                            'date': parts[0],
                            'message': parts[1],
                            'author': parts[2]
                        })
                        
    except Exception as e:
        print(f"Erro ao obter estatísticas: {e}")
        
    return stats

def get_component_info(repo_path: str) -> Dict[str, Any]:
    """Analisa componentes do sistema"""
    components = {
        'et_ultimate': {
            'path': 'opt/et_ultimate',
            'description': 'Sistema ET Ultimate - Cérebro Principal',
            'modules': [],
            'status': 'active'
        },
        'projetos': {
            'path': 'projetos',
            'description': 'Projetos Diversos',
            'modules': [],
            'status': 'active'
        },
        'ml': {
            'path': 'ml',
            'description': 'Machine Learning Models',
            'modules': [],
            'status': 'active'
        },
        'penin_omega': {
            'path': 'penin_omega',
            'description': 'Sistema PENIN Omega',
            'modules': [],
            'status': 'active'
        }
    }
    
    for comp_name, comp_info in components.items():
        comp_path = os.path.join(repo_path, comp_info['path'])
        
        if os.path.exists(comp_path):
            # Encontrar módulos Python
            for root, dirs, files in os.walk(comp_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        rel_path = os.path.relpath(os.path.join(root, file), comp_path)
                        components[comp_name]['modules'].append(rel_path)
                        
    return components

def load_sections(sections_dir: str) -> Dict[str, str]:
    """Carrega seções do README"""
    sections = {}
    
    if not os.path.exists(sections_dir):
        return sections
        
    for file in sorted(os.listdir(sections_dir)):
        if file.endswith('.md'):
            section_name = os.path.splitext(file)[0]
            file_path = os.path.join(sections_dir, file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sections[section_name] = f.read()
            except Exception as e:
                print(f"Erro ao carregar seção {file}: {e}")
                
    return sections

def build_readme(config: Dict[str, Any], repo) -> None:
    """Constrói o README mestre"""
    repo_path = repo.working_dir
    readme_config = config['readme']
    
    # Obter informações
    git_info = get_git_info(repo_path)
    repo_stats = get_repo_stats(repo_path)
    components = get_component_info(repo_path)
    sections = load_sections(readme_config['sections_dir'])
    
    # Configurar Jinja2
    template_dir = os.path.dirname(readme_config['template_path'])
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(os.path.basename(readme_config['template_path']))
    
    # Dados para o template
    template_data = {
        'project': readme_config['project'],
        'git_info': git_info,
        'repo_stats': repo_stats,
        'components': components,
        'sections': sections,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config
    }
    
    # Renderizar template
    readme_content = template.render(**template_data)
    
    # Salvar README
    output_path = readme_config['output_path']
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
        
    print(f"README gerado: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python build_readme.py <config.yaml> <repo_path>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    repo_path = sys.argv[2]
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Simular objeto repo
    class MockRepo:
        def __init__(self, path):
            self.working_dir = path
            
    repo = MockRepo(repo_path)
    build_readme(config, repo)