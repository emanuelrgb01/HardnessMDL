#!/usr/bin/env bash
set -e

PYTHON_VERSION=3.9
VENV_DIR_39=".venv39"

python$PYTHON_VERSION -m venv $VENV_DIR_39

source $VENV_DIR_39/bin/activate

pip install uv pip-tools

pip-compile requirements_39.in --output-file requirements_39.txt --upgrade

uv pip install -r requirements_39.txt

