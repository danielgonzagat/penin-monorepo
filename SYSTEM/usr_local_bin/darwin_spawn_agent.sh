#!/usr/bin/env bash
# Darwin Agent Spawner - Creates real agent from heritage
set -euo pipefail

NEWBORN_JSON="${1:-}"
AGENT_ID="${2:-}"

[ -z "$NEWBORN_JSON" ] && { echo "Usage: $0 /path/newborn.json [agent_id]"; exit 2; }
[ ! -f "$NEWBORN_JSON" ] && { echo "Heritage file not found: $NEWBORN_JSON"; exit 1; }

# Extract agent ID from heritage if not provided
if [ -z "$AGENT_ID" ]; then
    AGENT_ID=$(jq -r '.agent_id // empty' "$NEWBORN_JSON" 2>/dev/null || echo "agent_$(date +%s)")
fi

echo "🐣 Spawning agent: $AGENT_ID"

# Create agent directory structure
AGENT_DIR="/root/agents/$AGENT_ID"
mkdir -p "$AGENT_DIR"/{config,state,models,logs}

# Copy heritage
cp "$NEWBORN_JSON" "$AGENT_DIR/heritage.json"

# Create agent configuration from heritage
cat > "$AGENT_DIR/config/agent.json" <<EOF
{
  "agent_id": "$AGENT_ID",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "origin": "darwin_spawn",
  "heritage_path": "$NEWBORN_JSON",
  "status": "active",
  "capabilities": {
    "learning": true,
    "self_modification": true,
    "memory": true
  },
  "thresholds": {
    "delta_linf_min": 0.0,
    "caos_ratio_min": 1.0,
    "I_min": 0.60,
    "P_min": 0.01
  }
}
EOF

# Initialize agent state
cat > "$AGENT_DIR/state/current.json" <<EOF
{
  "agent_id": "$AGENT_ID",
  "generation": 1,
  "fitness": 0.5,
  "experience": 0,
  "last_decision": null,
  "metrics": {
    "delta_linf": 0.0,
    "caos_ratio": 1.0,
    "I": 0.60,
    "P": 0.01,
    "novelty": 0.0
  }
}
EOF

# Create agent activation script
cat > "$AGENT_DIR/activate.sh" <<'SCRIPT'
#!/bin/bash
AGENT_DIR="$(dirname "$0")"
AGENT_ID="$(basename "$AGENT_DIR")"
echo "Activating agent: $AGENT_ID"
# Add agent-specific activation logic here
export AGENT_ID
export AGENT_CONFIG="$AGENT_DIR/config/agent.json"
export AGENT_STATE="$AGENT_DIR/state/current.json"
echo "Agent $AGENT_ID activated at $(date)"
SCRIPT
chmod +x "$AGENT_DIR/activate.sh"

# Update agents manifest
MANIFEST="/root/agents/manifest.json"
if [ -f "$MANIFEST" ]; then
    # Add to existing manifest
    jq --arg id "$AGENT_ID" --arg dir "$AGENT_DIR" \
       '.agents += [{"id": $id, "directory": $dir, "status": "active"}]' \
       "$MANIFEST" > "$MANIFEST.tmp" && mv "$MANIFEST.tmp" "$MANIFEST"
else
    # Create new manifest
    cat > "$MANIFEST" <<EOF
{
  "version": "1.0",
  "agents": [
    {
      "id": "$AGENT_ID",
      "directory": "$AGENT_DIR",
      "status": "active"
    }
  ]
}
EOF
fi

echo "✅ Agent spawned successfully:"
echo "   Directory: $AGENT_DIR"
echo "   Config: $AGENT_DIR/config/agent.json"
echo "   State: $AGENT_DIR/state/current.json"
echo "   Activate: $AGENT_DIR/activate.sh"

exit 0