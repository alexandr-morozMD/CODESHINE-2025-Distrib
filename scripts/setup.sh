#!/usr/bin/env bash
set -euo pipefail

if [ -d ".venv" ]; then
  echo "[setup] Using existing virtual environment .venv"
else
  echo "[setup] Creating virtual environment .venv"
  python3 -m venv .venv
fi

. .venv/bin/activate

if ! pip --version >/dev/null 2>&1; then
  echo "[setup] pip not available in virtual environment"
  exit 1
fi

pip install --upgrade pip
pip install -r requirements.txt
