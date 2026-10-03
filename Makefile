.PHONY: setup models test test-all lint format docs docs-serve build demo preflight eval bench

setup:
	uv sync
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example; fill it in.")

models:
	uv run python scripts/download_models.py

test:
	uv run pytest -m "not models"

test-all:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff check --fix .
	uv run ruff format .

docs:
	uv run --group docs mkdocs build --strict

docs-serve:
	uv run --group docs mkdocs serve

build:
	rm -rf dist && uv build && uvx twine check --strict dist/*

demo:
	EARSHOT_DEMO=1 uv run band-demo demo

preflight:
	uv run band-demo preflight

# The band demo's own evaluation (needs your recordings in eval/data).
eval:
	uv run python eval/run_all.py
	uv run python scripts/update_readme_numbers.py
