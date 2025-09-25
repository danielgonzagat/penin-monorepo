#!/usr/bin/env bash
echo "[post_rollback] reason=$REASON ts=$TIMESTAMP success=${ROLLBACK_SUCCESS:-0}" >> /root/ia3_darwin_hooks.log
echo "✅ Rollback notificado: $REASON"
