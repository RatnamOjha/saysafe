"""Typer CLI entry point (`earshot ...`). Thin wrappers only; logic lives in the modules."""

import logging

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """earshot: voice-locked approvals and private replies for a screenless wearable."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@app.command("llm-ping")
def llm_ping(model: str = typer.Option("fast", help="fast or smart")) -> None:
    """Send one tiny request and print the model and latency."""
    from earshot import llm

    messages = [{"role": "user", "content": "Reply with the word pong."}]
    result = llm.complete(messages, model=model, timeout_s=5)
    if result is None:
        console.print("[red]No response.[/] Check LLM_API_KEY in .env (-v shows the reason).")
        raise typer.Exit(1)
    console.print(f"model={result.model}  latency={result.latency_ms:.0f} ms reply={result.text!r}")
