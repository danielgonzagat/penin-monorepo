#!/usr/bin/env bash
set -euo pipefail

OWNERS_FILE="/etc/et/owners"
[[ -f "$OWNERS_FILE" ]] || exit 0

# Carrega todos os donos
mapfile -t OWNERS <"$OWNERS_FILE"

check_cmd() {
    local cmd="\$*"
    for owner in "\${OWNERS[@]}"; do
        if [[ "\$cmd" =~ userdel[[:space:]]+\$owner ]] ||
           [[ "\$cmd" =~ passwd[[:space:]]+\$owner ]] ||
           [[ "\$cmd" =~ chsh[[:space:]]+\$owner ]] ||
           [[ "\$cmd" =~ chown.*\b\$owner\b ]]; then
            echo "[OwnerGuard] Bloqueado: tentativa de alterar/remover o dono (\$owner)"
            exit 1
        fi
    done
}

check_cmd "\$@"
# Se não bloquear, executa normalmente
exec "\$@"
