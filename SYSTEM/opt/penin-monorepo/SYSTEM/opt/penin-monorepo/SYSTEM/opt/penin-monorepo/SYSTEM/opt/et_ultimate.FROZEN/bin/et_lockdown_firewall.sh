#!/usr/bin/env bash
set -euo pipefail

USER_NAME="etbot"
USER_ID=$(id -u "$USER_NAME")

# Limpa regras antigas do nosso marcador
iptables -S | grep "owner UID match $USER_ID" >/dev/null 2>&1 && {
  iptables -D OUTPUT -m owner --uid-owner "$USER_ID" -j DROP || true
  iptables -D OUTPUT -m owner --uid-owner "$USER_ID" -o lo -j ACCEPT || true
}

# Permite apenas loopback para UID etbot; bloqueia todo o resto
iptables -A OUTPUT -m owner --uid-owner "$USER_ID" -o lo -j ACCEPT
iptables -A OUTPUT -m owner --uid-owner "$USER_ID" -j DROP

echo "[et_lockdown_firewall] Regras aplicadas para UID=$USER_ID (somente loopback)."
