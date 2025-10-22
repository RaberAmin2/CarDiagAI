#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "Python ist nicht installiert oder nicht im PATH verfügbar." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "Verwende Python-Interpreter: ${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt
