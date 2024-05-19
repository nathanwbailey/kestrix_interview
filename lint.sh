#!/bin/bash

EXCLUDE="venv"

echo "Running Black..."
black . --exclude "$EXCLUDE"

echo "Running isort..."
isort . --skip "$EXCLUDE"

echo "Running flake8..."
flake8 . --exclude "$EXCLUDE"

echo "Running mypy..."
mypy . --exclude "$EXCLUDE" --disable-error-code import-untyped

echo "Linting and formatting complete!"