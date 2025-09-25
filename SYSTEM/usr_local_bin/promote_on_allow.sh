#!/usr/bin/env bash
# promote_on_allow.sh — promoção/rollback atômico com verificação de hash e WORM

set -euo pipefail

CKPT=""
MODEL_DIR=""
SYMLINK=""
WORM_PATH=""
CHAMPION_PATH=""
DRY_RUN="false"
ROLLBACK="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CKPT="$2"; shift 2 ;;
    --model-dir)  MODEL_DIR="$2"; shift 2 ;;
    --symlink)    SYMLINK="$2"; shift 2 ;;
    --worm)       WORM_PATH="$2"; shift 2 ;;
    --champion)   CHAMPION_PATH="$2"; shift 2 ;;
    --dry-run)    DRY_RUN="true"; shift 1 ;;
    --rollback)   ROLLBACK="true"; shift 1 ;;
    *) echo "arg desconhecido: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${MODEL_DIR}" "$(dirname "${WORM_PATH}")"

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# -------- rollback branch --------
if [[ "${ROLLBACK:-false}" == "true" ]]; then
  if [[ -z "${CHAMPION_PATH}" ]]; then
    echo "rollback solicitado mas --champion não informado" >&2
    exit 3
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "{\"ts\":\"${TS}\",\"rollback\":\"true\",\"champion\":\"${CHAMPION_PATH}\",\"symlink\":\"${SYMLINK}\"}"
    printf 'ROLLBACK | %s | %s | champion=%s\n' "${TS}" "${SYMLINK}" "${CHAMPION_PATH}" >> "${WORM_PATH}"
    exit 0
  fi
  # rollback real: symlink volta ao champion
  TMP_LINK="${SYMLINK}.rollback.$$"
  ln -sfn "${CHAMPION_PATH}" "${TMP_LINK}"
  mv -Tf "${TMP_LINK}" "${SYMLINK}"
  printf 'ROLLBACK | %s | %s | champion=%s\n' "${TS}" "${SYMLINK}" "${CHAMPION_PATH}" >> "${WORM_PATH}"
  echo "{\"ts\":\"${TS}\",\"rollback\":\"true\",\"champion\":\"${CHAMPION_PATH}\",\"symlink\":\"${SYMLINK}\"}"
  exit 0
fi

# -------- promoção branch --------
if [[ -z "${CKPT}" || -z "${MODEL_DIR}" || -z "${SYMLINK}" || -z "${WORM_PATH}" ]]; then
  echo "uso inválido — faltam parâmetros obrigatórios" >&2
  exit 2
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "checkpoint não existe: ${CKPT}" >&2
  exit 3
fi

SHA=$(sha256sum "${CKPT}" | awk '{print $1}')
BASENAME="$(basename "${CKPT}")"
DEST="${MODEL_DIR}/${BASENAME}"

PROMO_JSON="{\"ts\":\"${TS}\",\"checkpoint\":\"${CKPT}\",\"dest\":\"${DEST}\",\"sha256\":\"${SHA}\",\"symlink\":\"${SYMLINK}\",\"dry_run\":${DRY_RUN}}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "${PROMO_JSON}"
  printf '%s | %s | prev=DRYRUN | %s\n' "${SHA}" "${TS}" "${PROMO_JSON}" >> "${WORM_PATH}"
  exit 0
fi

cp -f -- "${CKPT}" "${DEST}"
chmod 0640 "${DEST}"

TMP_LINK="${SYMLINK}.new.$$"
ln -sfn "${DEST}" "${TMP_LINK}"
mv -Tf "${TMP_LINK}" "${SYMLINK}"

printf '%s | %s | prev=PROMOTE | %s\n' "${SHA}" "${TS}" "${PROMO_JSON}" >> "${WORM_PATH}"
echo "${PROMO_JSON}"