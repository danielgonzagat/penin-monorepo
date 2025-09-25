#!/usr/bin/env python3
"""
Integração com a API de Agentes em Segundo Plano do Cursor
Permite criar e gerenciar agentes que trabalham automaticamente no repositório
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

class CursorAPIClient:
    """Cliente para interagir com a API do Cursor"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('CURSOR_API_KEY')
        if not self.api_key:
            raise ValueError("CURSOR_API_KEY não configurada")
        
        self.base_url = "https://api.cursor.com"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
    def create_agent(self, name: str, prompt: str, repository: str) -> Dict[str, Any]:
        """Cria um novo agente em segundo plano"""
        url = f"{self.base_url}/agents"
        
        data = {
            'name': name,
            'prompt': prompt,
            'repository': repository
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erro ao criar agente {name}: {e}")
            return None
            
    def list_agents(self) -> List[Dict[str, Any]]:
        """Lista todos os agentes ativos"""
        url = f"{self.base_url}/agents"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json().get('agents', [])
        except requests.exceptions.RequestException as e:
            print(f"Erro ao listar agentes: {e}")
            return []
            
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Obtém o status de um agente específico"""
        url = f"{self.base_url}/agents/{agent_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erro ao obter status do agente {agent_id}: {e}")
            return None
            
    def add_followup_prompt(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        """Adiciona uma instrução adicional a um agente em execução"""
        url = f"{self.base_url}/agents/{agent_id}/prompts"
        
        data = {'prompt': prompt}
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erro ao adicionar prompt ao agente {agent_id}: {e}")
            return None
            
    def delete_agent(self, agent_id: str) -> bool:
        """Remove um agente"""
        url = f"{self.base_url}/agents/{agent_id}"
        
        try:
            response = requests.delete(url, headers=self.headers)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Erro ao deletar agente {agent_id}: {e}")
            return False

class PENINAgentManager:
    """Gerenciador de agentes do sistema PENIN"""
    
    def __init__(self, github_user: str = "danielgonzagat", repo_name: str = "penin-monorepo"):
        self.github_user = github_user
        self.repo_name = repo_name
        self.repository = f"{github_user}/{repo_name}"
        self.client = CursorAPIClient()
        self.agents = {}
        
    def setup_default_agents(self):
        """Configura os agentes padrão do sistema PENIN"""
        
        default_agents = [
            {
                'name': 'penin-code-reviewer',
                'prompt': """
                You are an AI code reviewer for the PENIN system.
                Your tasks:
                1. Review all new code for quality, performance, and security
                2. Suggest improvements and best practices
                3. Ensure code follows Python PEP8 standards
                4. Check for potential bugs and edge cases
                5. Verify that documentation is up-to-date
                
                Focus areas:
                - ET Ultimate (brain system)
                - Machine Learning models
                - PENIN Omega (evolution system)
                - Integration and compatibility
                
                Always provide constructive feedback and concrete suggestions.
                """
            },
            {
                'name': 'penin-bug-fixer',
                'prompt': """
                You are an AI bug fixer for the PENIN system.
                Your tasks:
                1. Automatically detect and fix bugs in the codebase
                2. Fix import errors and missing dependencies
                3. Resolve type mismatches and logic errors
                4. Fix broken tests and ensure all tests pass
                5. Handle exceptions and error cases properly
                
                Priority areas:
                - Critical system failures
                - Data integrity issues
                - Security vulnerabilities
                - Performance bottlenecks
                
                Always test your fixes before committing.
                """
            },
            {
                'name': 'penin-documentation-updater',
                'prompt': """
                You are an AI documentation maintainer for the PENIN system.
                Your tasks:
                1. Keep README files updated with latest changes
                2. Update docstrings when functions change
                3. Maintain the master README with system evolution
                4. Document new features and components
                5. Create examples and tutorials for complex features
                
                Documentation standards:
                - Clear and concise explanations
                - Code examples where appropriate
                - Updated API references
                - Changelog maintenance
                
                Ensure documentation is always synchronized with code.
                """
            },
            {
                'name': 'penin-security-scanner',
                'prompt': """
                You are an AI security analyst for the PENIN system.
                Your tasks:
                1. Scan for security vulnerabilities
                2. Check for exposed secrets and credentials
                3. Verify input validation and sanitization
                4. Check for SQL injection and XSS vulnerabilities
                5. Ensure secure communication and data storage
                
                Security priorities:
                - API key protection
                - User data privacy
                - Secure authentication
                - Encryption of sensitive data
                
                Report all findings with severity levels and fixes.
                """
            },
            {
                'name': 'penin-evolution-agent',
                'prompt': """
                You are the evolution agent for the PENIN system.
                Your tasks:
                1. Identify opportunities for system improvement
                2. Suggest new features based on usage patterns
                3. Optimize existing code for better performance
                4. Implement self-improvement mechanisms
                5. Track system evolution from zero to SOTA
                
                Evolution strategies:
                - Incremental improvements
                - Feature expansion
                - Performance optimization
                - Architecture refinement
                
                Document all evolution steps in the master README.
                """
            }
        ]
        
        print("🤖 Configurando agentes padrão do sistema PENIN...")
        
        for agent_config in default_agents:
            agent = self.client.create_agent(
                name=agent_config['name'],
                prompt=agent_config['prompt'],
                repository=self.repository
            )
            
            if agent:
                self.agents[agent_config['name']] = agent
                print(f"  ✓ Agente {agent_config['name']} criado com sucesso")
            else:
                print(f"  ✗ Erro ao criar agente {agent_config['name']}")
                
        return self.agents
        
    def monitor_agents(self):
        """Monitora o status dos agentes ativos"""
        print("\n📊 Status dos Agentes PENIN:")
        print("-" * 50)
        
        agents = self.client.list_agents()
        
        if not agents:
            print("Nenhum agente ativo no momento")
            return
            
        for agent in agents:
            if 'penin' in agent.get('name', '').lower():
                status = self.client.get_agent_status(agent['id'])
                if status:
                    print(f"\n🤖 {agent['name']}")
                    print(f"  ID: {agent['id']}")
                    print(f"  Status: {status.get('status', 'unknown')}")
                    print(f"  Criado: {status.get('created_at', 'N/A')}")
                    print(f"  Última atividade: {status.get('last_activity', 'N/A')}")
                    
    def trigger_code_review(self, commit_hash: str = None):
        """Dispara revisão de código para um commit específico"""
        prompt = f"Review the latest changes in the repository"
        if commit_hash:
            prompt = f"Review the changes in commit {commit_hash}"
            
        reviewer = self.agents.get('penin-code-reviewer')
        if reviewer:
            self.client.add_followup_prompt(reviewer['id'], prompt)
            print(f"✓ Revisão de código disparada para {commit_hash or 'últimas mudanças'}")
            
    def trigger_bug_fix(self, issue_description: str):
        """Dispara correção automática de bug"""
        prompt = f"Fix the following issue: {issue_description}"
        
        fixer = self.agents.get('penin-bug-fixer')
        if fixer:
            self.client.add_followup_prompt(fixer['id'], prompt)
            print(f"✓ Correção de bug disparada: {issue_description}")
            
    def trigger_documentation_update(self):
        """Dispara atualização de documentação"""
        prompt = "Update all documentation to reflect the latest code changes"
        
        updater = self.agents.get('penin-documentation-updater')
        if updater:
            self.client.add_followup_prompt(updater['id'], prompt)
            print("✓ Atualização de documentação disparada")
            
    def cleanup_agents(self):
        """Remove todos os agentes PENIN"""
        print("\n🧹 Removendo agentes PENIN...")
        
        agents = self.client.list_agents()
        for agent in agents:
            if 'penin' in agent.get('name', '').lower():
                if self.client.delete_agent(agent['id']):
                    print(f"  ✓ Agente {agent['name']} removido")
                else:
                    print(f"  ✗ Erro ao remover agente {agent['name']}")

def main():
    """Função principal para teste e configuração"""
    
    print("=" * 60)
    print("   PENIN - Integração com Cursor API")
    print("   Sistema de Agentes em Segundo Plano")
    print("=" * 60)
    print()
    
    # Verificar API key
    if not os.getenv('CURSOR_API_KEY'):
        print("⚠️  CURSOR_API_KEY não configurada!")
        print("Configure com: export CURSOR_API_KEY='sua_chave_aqui'")
        print()
        print("Para obter sua chave:")
        print("1. Acesse https://cursor.com/dashboard")
        print("2. Vá para Integrations → API Keys")
        print("3. Crie uma nova chave")
        return
        
    # Criar gerenciador
    manager = PENINAgentManager()
    
    # Menu de opções
    while True:
        print("\n" + "=" * 40)
        print("Escolha uma opção:")
        print("1. Configurar agentes padrão")
        print("2. Monitorar status dos agentes")
        print("3. Disparar revisão de código")
        print("4. Disparar correção de bug")
        print("5. Disparar atualização de documentação")
        print("6. Limpar todos os agentes")
        print("0. Sair")
        print("=" * 40)
        
        choice = input("\nOpção: ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            manager.setup_default_agents()
        elif choice == '2':
            manager.monitor_agents()
        elif choice == '3':
            commit = input("Commit hash (deixe vazio para últimas mudanças): ").strip()
            manager.trigger_code_review(commit if commit else None)
        elif choice == '4':
            issue = input("Descrição do bug: ").strip()
            if issue:
                manager.trigger_bug_fix(issue)
        elif choice == '5':
            manager.trigger_documentation_update()
        elif choice == '6':
            confirm = input("Confirmar remoção de todos os agentes? (s/n): ").strip()
            if confirm.lower() == 's':
                manager.cleanup_agents()
        else:
            print("Opção inválida!")
            
    print("\n✓ Encerrando integração com Cursor API")

if __name__ == "__main__":
    main()