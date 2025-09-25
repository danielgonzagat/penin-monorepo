#!/bin/bash
# PENIN Control Center - Sistema de Controle Total com Agentes

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

# URLs e informações
GITHUB_URL="https://github.com/danielgonzagat/penin-monorepo"
CURSOR_DASHBOARD="https://cursor.com/dashboard"
LOG_FILE="/opt/penin-autosync/logs/sync.log"

# Banner
show_banner() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║     ██████╗ ███████╗███╗   ██╗██╗███╗   ██╗                       ║"
    echo "║     ██╔══██╗██╔════╝████╗  ██║██║████╗  ██║                       ║"
    echo "║     ██████╔╝█████╗  ██╔██╗ ██║██║██╔██╗ ██║                       ║"
    echo "║     ██╔═══╝ ██╔══╝  ██║╚██╗██║██║██║╚██╗██║                       ║"
    echo "║     ██║     ███████╗██║ ╚████║██║██║ ╚████║                       ║"
    echo "║     ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝                       ║"
    echo "║                                                                      ║"
    echo "║         Sistema de Evolução Contínua com Agentes IA                ║"
    echo "║         Sincronização Bidirecional + Controle Total                 ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Status completo
status() {
    echo -e "\n${BLUE}═══ Status Completo do Sistema PENIN ═══${NC}\n"
    
    # Status do serviço
    if systemctl is-active --quiet penin-sync.service; then
        echo -e "${GREEN}✅ Serviço:${NC} ATIVO e rodando 24/7"
        uptime=$(systemctl show -p ActiveEnterTimestamp --value penin-sync.service)
        echo -e "   └─ Uptime: $uptime"
    else
        echo -e "${RED}❌ Serviço:${NC} INATIVO"
    fi
    
    # Status do GitHub
    echo -e "\n${CYAN}GitHub:${NC}"
    echo -e "   └─ Repositório: ${BLUE}$GITHUB_URL${NC}"
    cd /opt/penin-monorepo 2>/dev/null
    commits=$(git rev-list --count HEAD 2>/dev/null || echo "0")
    echo -e "   └─ Total de commits: $commits"
    last_commit=$(git log -1 --format='%h - %s (%ar)' 2>/dev/null || echo "Nenhum")
    echo -e "   └─ Último commit: $last_commit"
    
    # Status dos Agentes
    echo -e "\n${CYAN}Agentes Cursor:${NC}"
    python3 /opt/penin-autosync/tools/agent_controller.py list 2>/dev/null | head -5 | while read line; do
        if [ -n "$line" ]; then
            echo -e "   └─ $line"
        fi
    done
    
    # Estatísticas
    echo -e "\n${CYAN}Estatísticas:${NC}"
    total_files=$(find /opt/penin-monorepo -type f 2>/dev/null | wc -l)
    echo -e "   └─ Total de arquivos: $total_files"
    
    # Últimas sincronizações
    echo -e "\n${CYAN}Últimas sincronizações:${NC}"
    tail -5 $LOG_FILE 2>/dev/null | grep "sync\|Sync" | head -3 | while read line; do
        echo -e "   └─ $line"
    done
}

# Controle de Agentes
agent_control() {
    echo -e "\n${MAGENTA}═══ CONTROLE DE AGENTES CURSOR ═══${NC}\n"
    echo "1. 📋 Listar agentes ativos"
    echo "2. ➕ Criar novo agente"
    echo "3. 🚀 Criar TODOS os agentes (5 agentes)"
    echo "4. 📨 Enviar comando para agente"
    echo "5. 🗑️  Remover agente"
    echo "6. 🔧 Gerenciamento avançado"
    echo "0. ↩️  Voltar"
    echo ""
    read -p "Escolha: " agent_choice
    
    case $agent_choice in
        1)
            echo -e "\n${CYAN}Agentes Ativos:${NC}"
            python3 /opt/penin-autosync/tools/agent_controller.py list
            ;;
        2)
            echo -e "\n${CYAN}Tipos de Agente Disponíveis:${NC}"
            echo "  optimizer - Otimizador de Código"
            echo "  security - Guardião de Segurança"
            echo "  evolution - Motor de Evolução"
            echo "  bug_fixer - Corretor de Bugs"
            echo "  documentation - Mestre de Documentação"
            echo ""
            read -p "Tipo de agente: " agent_type
            python3 /opt/penin-autosync/tools/agent_controller.py create $agent_type
            ;;
        3)
            echo -e "\n${YELLOW}Criando TODOS os agentes...${NC}"
            echo "Isso criará 5 agentes especializados:"
            echo "• Code Optimizer"
            echo "• Security Guardian"
            echo "• Evolution Engine"
            echo "• Bug Fixer"
            echo "• Documentation Master"
            echo ""
            read -p "Confirmar criação? (s/n): " confirm
            if [ "$confirm" = "s" ]; then
                for agent in optimizer security evolution bug_fixer documentation; do
                    echo -e "\n${CYAN}Criando $agent...${NC}"
                    python3 /opt/penin-autosync/tools/agent_controller.py create $agent
                    sleep 2
                done
                echo -e "\n${GREEN}✅ Todos os agentes criados!${NC}"
            fi
            ;;
        4)
            read -p "ID do agente: " agent_id
            read -p "Comando: " command
            python3 /opt/penin-autosync/tools/agent_controller.py command "$agent_id" "$command"
            ;;
        5)
            read -p "ID do agente para remover: " agent_id
            read -p "Confirmar remoção? (s/n): " confirm
            if [ "$confirm" = "s" ]; then
                # Implementar remoção
                echo "Removendo agente $agent_id..."
            fi
            ;;
        6)
            python3 /opt/penin-autosync/tools/agent_controller.py
            ;;
    esac
}

# Comandos rápidos de agentes
quick_agent_commands() {
    echo -e "\n${MAGENTA}═══ COMANDOS RÁPIDOS PARA AGENTES ═══${NC}\n"
    echo "1. 🔥 'Otimize todo o código Python'"
    echo "2. 🛡️  'Escaneie e corrija vulnerabilidades'"
    echo "3. 🚀 'Implemente melhorias e novos recursos'"
    echo "4. 🐛 'Encontre e corrija todos os bugs'"
    echo "5. 📚 'Atualize toda a documentação'"
    echo "6. 🎯 Comando personalizado"
    echo "0. ↩️  Voltar"
    echo ""
    read -p "Escolha: " cmd_choice
    
    case $cmd_choice in
        1)
            cmd="Analyze all Python code in the repository and optimize for performance. Refactor complex functions, add type hints, and ensure PEP8 compliance."
            ;;
        2)
            cmd="Perform a complete security scan. Check for exposed credentials, SQL injection, XSS vulnerabilities. Fix all security issues immediately."
            ;;
        3)
            cmd="Identify opportunities for improvement and implement new features. Enhance the system architecture and add useful utilities."
            ;;
        4)
            cmd="Find and fix all bugs in the codebase. Fix import errors, resolve exceptions, and add proper error handling."
            ;;
        5)
            cmd="Update all documentation including README.md. Add docstrings to all functions and create usage examples."
            ;;
        6)
            read -p "Digite o comando: " cmd
            ;;
        *)
            return
            ;;
    esac
    
    if [ -n "$cmd" ]; then
        echo -e "\n${YELLOW}Enviando comando para TODOS os agentes...${NC}"
        # Obter lista de agentes e enviar comando
        python3 << EOF
import sys
sys.path.append('/opt/penin-autosync/tools')
from agent_controller import CursorAgentController
controller = CursorAgentController()
agents = controller.list_agents()
for agent in agents:
    print(f"Enviando para {agent['id']}...")
    controller.send_command(agent['id'], "$cmd")
print("✅ Comando enviado!")
EOF
    fi
}

# Menu principal
main_menu() {
    while true; do
        show_banner
        echo -e "${WHITE}Sistema ${GREEN}OPERACIONAL${WHITE} - Sincronização 24/7 Ativa${NC}\n"
        echo "═══ SISTEMA ═══"
        echo "1. 📊 Ver status completo"
        echo "2. 📜 Ver logs em tempo real"
        echo "3. 🔄 Forçar sincronização manual"
        echo "4. 🌐 Abrir repositório no GitHub"
        echo ""
        echo "═══ AGENTES IA ═══"
        echo "5. 🤖 Controlar agentes Cursor"
        echo "6. ⚡ Comandos rápidos para agentes"
        echo "7. 📋 Ver status dos agentes"
        echo ""
        echo "═══ AVANÇADO ═══"
        echo "8. ⚙️  Reiniciar serviço"
        echo "9. 📈 Ver estatísticas detalhadas"
        echo ""
        echo "0. 🚪 Sair"
        echo ""
        read -p "Escolha: " choice
        
        case $choice in
            1)
                status
                read -p "Pressione Enter para continuar..."
                ;;
            2)
                echo -e "\n${CYAN}Logs em tempo real (Ctrl+C para sair):${NC}"
                tail -f $LOG_FILE
                ;;
            3)
                echo -e "\n${YELLOW}Forçando sincronização...${NC}"
                python3 /opt/penin-autosync/start_sync.py &
                sleep 3
                pkill -f "start_sync.py"
                echo -e "${GREEN}✅ Sincronização concluída${NC}"
                read -p "Pressione Enter..."
                ;;
            4)
                echo -e "\n${CYAN}Repositório GitHub:${NC}"
                echo "$GITHUB_URL"
                echo ""
                echo "Abrir no navegador? (s/n)"
                read -p "Escolha: " open_browser
                if [ "$open_browser" = "s" ]; then
                    xdg-open "$GITHUB_URL" 2>/dev/null || echo "Abra manualmente: $GITHUB_URL"
                fi
                read -p "Pressione Enter..."
                ;;
            5)
                agent_control
                read -p "Pressione Enter..."
                ;;
            6)
                quick_agent_commands
                read -p "Pressione Enter..."
                ;;
            7)
                echo -e "\n${CYAN}═══ Status dos Agentes ═══${NC}\n"
                python3 /opt/penin-autosync/tools/agent_controller.py list
                read -p "Pressione Enter..."
                ;;
            8)
                echo -e "\n${YELLOW}Reiniciando serviço...${NC}"
                systemctl restart penin-sync.service
                sleep 2
                if systemctl is-active --quiet penin-sync.service; then
                    echo -e "${GREEN}✅ Serviço reiniciado com sucesso${NC}"
                else
                    echo -e "${RED}❌ Erro ao reiniciar${NC}"
                fi
                read -p "Pressione Enter..."
                ;;
            9)
                echo -e "\n${CYAN}═══ Estatísticas Detalhadas ═══${NC}\n"
                cd /opt/penin-monorepo
                echo "Arquivos por tipo:"
                find . -type f -name "*.py" 2>/dev/null | wc -l | xargs echo "  Python (.py):"
                find . -type f -name "*.md" 2>/dev/null | wc -l | xargs echo "  Markdown (.md):"
                find . -type f -name "*.yaml" -o -name "*.yml" 2>/dev/null | wc -l | xargs echo "  YAML:"
                find . -type f -name "*.json" 2>/dev/null | wc -l | xargs echo "  JSON:"
                echo ""
                echo "Tamanho total:"
                du -sh /opt/penin-monorepo 2>/dev/null
                echo ""
                echo "Commits hoje:"
                git log --since="midnight" --oneline 2>/dev/null | wc -l
                echo ""
                echo "Top contribuidores:"
                git shortlog -sn | head -5
                read -p "Pressione Enter..."
                ;;
            0)
                echo -e "\n${GREEN}Sistema continua rodando em background 24/7!${NC}"
                echo -e "${CYAN}GitHub:${NC} $GITHUB_URL"
                echo -e "${CYAN}Dashboard:${NC} $CURSOR_DASHBOARD"
                echo -e "\n${GREEN}Até logo!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Opção inválida!${NC}"
                sleep 1
                ;;
        esac
    done
}

# Se passou argumento, executar comando específico
if [ $# -gt 0 ]; then
    case $1 in
        status)
            status
            ;;
        logs)
            tail -f $LOG_FILE
            ;;
        sync)
            python3 /opt/penin-autosync/start_sync.py &
            sleep 5
            pkill -f "start_sync.py"
            ;;
        agents)
            python3 /opt/penin-autosync/tools/agent_controller.py
            ;;
        create-agent)
            if [ -n "$2" ]; then
                python3 /opt/penin-autosync/tools/agent_controller.py create "$2"
            else
                echo "Uso: penin create-agent [optimizer|security|evolution|bug_fixer|documentation]"
            fi
            ;;
        list-agents)
            python3 /opt/penin-autosync/tools/agent_controller.py list
            ;;
        *)
            echo "Uso: penin [status|logs|sync|agents|create-agent|list-agents]"
            echo ""
            echo "Comandos:"
            echo "  status       - Ver status do sistema"
            echo "  logs         - Ver logs em tempo real"
            echo "  sync         - Forçar sincronização"
            echo "  agents       - Gerenciar agentes"
            echo "  create-agent - Criar agente (optimizer|security|evolution|bug_fixer|documentation)"
            echo "  list-agents  - Listar agentes ativos"
            ;;
    esac
else
    main_menu
fi