#!/usr/bin/env bash
# Wrapper around enrich.py for scheduled (launchd) execution.
#
# - Launches Anki if it isn't already running (AnkiConnect needs it).
# - Runs enrich.py via the project's virtualenv Python.
# - Appends all output to logs/enrich.log with a timestamp banner.

set -uo pipefail

PROJECT_DIR="/Users/theo/Projects/korean_auto"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/enrich.log"
PY="${PROJECT_DIR}/.venv/bin/python"

mkdir -p "${LOG_DIR}"

{
  echo
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') :: enrich run starting ====="

  # Ensure Anki is running. `open -a Anki` is a no-op if it's already up,
  # but we sleep only when we actually had to launch it so AnkiConnect has
  # time to come online.
  if ! pgrep -x Anki >/dev/null; then
    echo "[wrapper] Anki not running — launching it"
    open -a Anki
    sleep 25
  else
    echo "[wrapper] Anki already running"
  fi

  cd "${PROJECT_DIR}"
  "${PY}" enrich.py
  EXIT=$?
  echo "[wrapper] enrich.py exited with code ${EXIT}"
  echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') :: enrich run finished ====="
} >> "${LOG_FILE}" 2>&1
