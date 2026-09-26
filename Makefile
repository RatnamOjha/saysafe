.PHONY: setup models test test-all lint format demo preflight eval video

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

format:
	uv run ruff check --fix .
	uv run ruff format .

demo:
	EARSHOT_DEMO=1 uv run earshot demo

preflight:
	uv run earshot preflight

eval:
	@echo "eval: not built yet (E3)"

video:
	@echo "video: not built yet (G1)"
