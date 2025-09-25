#!/usr/bin/env python3
"""
Controlador de Agentes Cursor - Interface Completa
Permite criar, gerenciar e comandar agentes diretamente
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AgentController')

class CursorAgentController:
    def __init__(self):
        self.api_key = "key_4041137deb2a6db7c18be16dd59a13a8f3b0a1a04bae91b5b100adb060881644"
        self.base_url = "https://api.cursor.com"
        self.github_repo = "https://github.com/danielgonzagat/penin-monorepo"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
    def test_connection(self) -> bool:
        """Testa conexão com API"""
        try:
            response = requests.get(
                f"{self.base_url}/v0/me",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Conexão com API Cursor estabelecida!")
                return True
            else:
                print(f"❌ Erro de autenticação: {response.status_code}")
                print(f"Resposta: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Erro ao conectar: {e}")
            return False
            
    def list_agents(self) -> List[Dict]:
        """Lista todos os agentes ativos"""
        try:
            response = requests.get(
                f"{self.base_url}/v0/agents",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                agents = response.json().get('agents', [])
                return agents
            else:
                print(f"Erro ao listar agentes: {response.status_code}")
                return []
        except Exception as e:
            print(f"Erro: {e}")
            return []
            
    def create_agent(self, name: str, prompt: str, auto_start: bool = True) -> Optional[Dict]:
        """Cria um novo agente"""
        try:
            data = {
                'prompt': {
                    'text': prompt
                },
                'source': {
                    'repository': self.github_repo,
                    'ref': 'main'
                }
            }
            
            print(f"🤖 Criando agente: {name}...")
            
            response = requests.post(
                f"{self.base_url}/v0/agents",
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                agent = response.json()
                print(f"✅ Agente criado com sucesso!")
                print(f"   ID: {agent.get('id')}")
                print(f"   Status: {agent.get('status')}")
                return agent
            else:
                print(f"❌ Erro ao criar agente: {response.status_code}")
                print(f"Resposta: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            return None
            
    def send_command(self, agent_id: str, command: str) -> bool:
        """Envia comando/followup para agente"""
        try:
            data = {
                'prompt': {
                    'text': command
                }
            }
            
            response = requests.post(
                f"{self.base_url}/v0/agents/{agent_id}/followup",
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ Comando enviado para agente {agent_id}")
                return True
            else:
                print(f"❌ Erro ao enviar comando: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
            
    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """Obtém status detalhado do agente"""
        try:
            response = requests.get(
                f"{self.base_url}/v0/agents/{agent_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            print(f"Erro: {e}")
            return None
            
    def delete_agent(self, agent_id: str) -> bool:
        """Remove um agente"""
        try:
            response = requests.delete(
                f"{self.base_url}/v0/agents/{agent_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                print(f"✅ Agente {agent_id} removido")
                return True
            else:
                print(f"❌ Erro ao remover agente")
                return False
                
        except Exception as e:
            print(f"Erro: {e}")
            return False

# Agentes pré-configurados
PRESET_AGENTS = {
    "optimizer": {
        "name": "Code Optimizer",
        "prompt": """You are the PENIN Code Optimizer Agent.
        Repository: https://github.com/danielgonzagat/penin-monorepo
        
        Your continuous tasks:
        1. Analyze all Python code and optimize for performance
        2. Refactor complex functions into cleaner code
        3. Add type hints and improve documentation
        4. Ensure PEP8 compliance
        5. Remove duplicate code and improve DRY principle
        
        Work autonomously and commit improvements with clear messages."""
    },
    "security": {
        "name": "Security Guardian",
        "prompt": """You are the PENIN Security Guardian Agent.
        Repository: https://github.com/danielgonzagat/penin-monorepo
        
        Your security tasks:
        1. Scan for security vulnerabilities
        2. Check for exposed credentials or API keys
        3. Fix SQL injection and XSS vulnerabilities
        4. Update vulnerable dependencies
        5. Implement security best practices
        
        Fix all security issues immediately with detailed explanations."""
    },
    "evolution": {
        "name": "Evolution Engine",
        "prompt": """You are the PENIN Evolution Engine Agent.
        Repository: https://github.com/danielgonzagat/penin-monorepo
        
        Your evolution tasks:
        1. Identify opportunities for new features
        2. Implement improvements based on code patterns
        3. Add useful utilities and helper functions
        4. Enhance system architecture
        5. Document evolution in README
        
        Make the system better with each iteration."""
    },
    "bug_fixer": {
        "name": "Bug Fixer",
        "prompt": """You are the PENIN Bug Fixer Agent.
        Repository: https://github.com/danielgonzagat/penin-monorepo
        
        Your debugging tasks:
        1. Find and fix all bugs in the code
        2. Fix import errors and circular dependencies
        3. Resolve type errors and exceptions
        4. Add error handling where missing
        5. Create unit tests for critical functions
        
        Test all fixes before committing."""
    },
    "documentation": {
        "name": "Documentation Master",
        "prompt": """You are the PENIN Documentation Master Agent.
        Repository: https://github.com/danielgonzagat/penin-monorepo
        
        Your documentation tasks:
        1. Keep README.md always updated
        2. Add docstrings to all functions and classes
        3. Create usage examples for complex features
        4. Document API endpoints and interfaces
        5. Maintain CHANGELOG.md with all changes
        
        Ensure documentation is clear and helpful."""
    }
}

def interactive_menu():
    """Menu interativo para controle de agentes"""
    controller = CursorAgentController()
    
    while True:
        print("\n" + "="*60)
        print("   CONTROLE DE AGENTES CURSOR - PENIN")
        print("="*60)
        print("\n1. 🔌 Testar conexão com API")
        print("2. 📋 Listar agentes ativos")
        print("3. ➕ Criar novo agente (preset)")
        print("4. 🎯 Criar agente customizado")
        print("5. 📨 Enviar comando para agente")
        print("6. 📊 Ver status de agente")
        print("7. 🗑️  Remover agente")
        print("8. 🚀 Criar TODOS os agentes preset")
        print("9. 🔄 Atualizar agentes com novo comando")
        print("0. 🚪 Voltar")
        
        choice = input("\nEscolha: ").strip()
        
        if choice == '0':
            break
            
        elif choice == '1':
            controller.test_connection()
            
        elif choice == '2':
            print("\n📋 Agentes Ativos:")
            agents = controller.list_agents()
            if agents:
                for agent in agents:
                    print(f"\n🤖 ID: {agent.get('id')}")
                    print(f"   Status: {agent.get('status')}")
                    print(f"   Criado: {agent.get('created_at')}")
            else:
                print("Nenhum agente ativo")
                
        elif choice == '3':
            print("\n🎯 Escolha um preset:")
            for key, preset in PRESET_AGENTS.items():
                print(f"  {key}: {preset['name']}")
            
            preset_key = input("\nPreset: ").strip()
            if preset_key in PRESET_AGENTS:
                preset = PRESET_AGENTS[preset_key]
                agent = controller.create_agent(preset['name'], preset['prompt'])
            else:
                print("Preset inválido")
                
        elif choice == '4':
            name = input("Nome do agente: ").strip()
            print("Digite o prompt (termine com linha vazia):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            prompt = '\n'.join(lines)
            
            if prompt:
                controller.create_agent(name, prompt)
                
        elif choice == '5':
            agent_id = input("ID do agente: ").strip()
            command = input("Comando: ").strip()
            if agent_id and command:
                controller.send_command(agent_id, command)
                
        elif choice == '6':
            agent_id = input("ID do agente: ").strip()
            if agent_id:
                status = controller.get_agent_status(agent_id)
                if status:
                    print(f"\n📊 Status do Agente {agent_id}:")
                    print(json.dumps(status, indent=2))
                else:
                    print("Agente não encontrado")
                    
        elif choice == '7':
            agent_id = input("ID do agente para remover: ").strip()
            if agent_id:
                confirm = input(f"Confirmar remoção de {agent_id}? (s/n): ").strip()
                if confirm.lower() == 's':
                    controller.delete_agent(agent_id)
                    
        elif choice == '8':
            print("\n🚀 Criando TODOS os agentes preset...")
            for key, preset in PRESET_AGENTS.items():
                print(f"\nCriando {preset['name']}...")
                controller.create_agent(preset['name'], preset['prompt'])
                time.sleep(2)  # Evitar rate limiting
            print("\n✅ Todos os agentes criados!")
            
        elif choice == '9':
            command = input("Comando para TODOS os agentes: ").strip()
            if command:
                agents = controller.list_agents()
                for agent in agents:
                    print(f"Enviando para {agent['id']}...")
                    controller.send_command(agent['id'], command)
                    time.sleep(1)
                print("✅ Comando enviado para todos!")
        
        input("\nPressione Enter para continuar...")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        controller = CursorAgentController()
        
        if sys.argv[1] == "test":
            controller.test_connection()
            
        elif sys.argv[1] == "list":
            agents = controller.list_agents()
            for agent in agents:
                print(f"{agent.get('id')}: {agent.get('status')}")
                
        elif sys.argv[1] == "create" and len(sys.argv) > 2:
            preset_key = sys.argv[2]
            if preset_key in PRESET_AGENTS:
                preset = PRESET_AGENTS[preset_key]
                controller.create_agent(preset['name'], preset['prompt'])
                
        elif sys.argv[1] == "command" and len(sys.argv) > 3:
            agent_id = sys.argv[2]
            command = ' '.join(sys.argv[3:])
            controller.send_command(agent_id, command)
            
        else:
            print("Uso: agent_controller.py [test|list|create <preset>|command <id> <cmd>]")
    else:
        interactive_menu()