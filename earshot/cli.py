"""Typer CLI entry point (`earshot ...`). Thin wrappers only; logic lives in the modules."""

import logging
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """earshot: voice-locked approvals and private replies for a screenless wearable."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    if not verbose:
        _quiet_libraries()


def _quiet_libraries() -> None:
    """Model libraries print loading chatter and deprecation warnings; hide them."""
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for name in ("speechbrain", "faster_whisper", "httpx", "piper"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _load_voice_models() -> None:
    from earshot.audio import vad
    from earshot.identity.embed import _encoder

    with console.status("Loading voice model..."):
        vad.segments(np.zeros(16000, dtype=np.float32))
        _encoder()


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


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/]")
    raise typer.Exit(1)


@app.command()
def record(
    seconds: float = typer.Option(3.0, "--seconds", "-s"),
    out: Path = typer.Option(Path("recording.wav"), "--out", "-o"),
) -> None:
    """Record from the mic to a wav file."""
    from earshot.audio import capture
    from earshot.audio.io import rms_dbfs, save_wav

    console.print(f"Recording {seconds:.1f} s from [bold]{capture.input_device_name()}[/]...")
    audio = capture.record(seconds)
    save_wav(out, audio)
    console.print(f"Saved {out} ({rms_dbfs(audio):.1f} dBFS)")


@app.command()
def transcribe(
    path: Path | None = typer.Argument(None, help="Audio file. Omit with --live."),
    live: bool = typer.Option(False, "--live", help="Record 3 s from the mic instead."),
    purpose: str = typer.Option("command", help="command (small.en) or reply (base.en)"),
) -> None:
    """Transcribe a file or a 3 s mic clip, with latency."""
    from earshot.audio import capture, io, stt

    if live:
        console.print("Speak now (3 s)...")
        audio = capture.record(3.0)
    elif path:
        audio = io.load_audio(path)
    else:
        _fail("Give a file path or --live.")
    t = stt.get_stt(purpose).transcribe(audio)
    console.print(f"[bold]{t.text!r}[/]  ({t.model}, {t.latency_ms:.0f} ms)")


@app.command()
def say(text: str, volume: float = typer.Option(1.0, min=0.0, max=1.0)) -> None:
    """Speak text through Piper."""
    from earshot.audio.tts import get_tts

    tts = get_tts()
    audio = tts.synthesize(text)
    console.print(f"Synthesized in {tts.last_latency_ms:.0f} ms")
    tts.play(audio, volume)


@app.command()
def enroll(name: str = typer.Option(..., "--name")) -> None:
    """Guided voice enrollment: 8 short clips."""
    from earshot.audio import capture
    from earshot.identity.enroll import run_enrollment
    from earshot.identity.profile_store import ProfileError, ProfileStore

    try:
        store = ProfileStore()
    except ProfileError as e:
        _fail(str(e))
    mic = capture.input_device_name()
    console.print(f"Enrolling [bold]{name}[/] with mic [bold]{mic}[/].")
    console.print("Use the room and mic you'll demo with. Speak normally. Each clip is 3 s.\n")
    _load_voice_models()
    console.print("For each clip: read the line, press Enter, then say it [bold]out loud[/].\n")

    def ready(prompt: str) -> None:
        console.print(f"[bold cyan]{prompt}[/]")
        typer.prompt("  Press Enter when ready", default="", show_default=False)

    profile = run_enrollment(name, capture.record, mic, show=console.print, ready=ready)
    path = store.save(profile)
    console.print(f"\n[green]Saved encrypted profile[/] {path}")
    console.print(f"Median speech level: {profile.median_rms_dbfs:.1f} dBFS")


@app.command()
def verify(
    name: str = typer.Option(..., "--name"),
    live: bool = typer.Option(False, "--live", help="Record from the mic."),
    count: int = typer.Option(5, "--count", "-n", help="Live attempts."),
    file: list[Path] = typer.Option(None, "--file", "-f", help="Score audio files instead."),
) -> None:
    """Score speech against an enrolled profile."""
    from rich.table import Table

    from earshot.audio import capture, io
    from earshot.identity import verify as v
    from earshot.identity.profile_store import ProfileError, ProfileStore

    try:
        profile = ProfileStore().load(name)
    except ProfileError as e:
        _fail(str(e))
    t = v.thresholds()
    if not live and not file:
        _fail("Use --live or --file.")
    _load_voice_models()

    table = Table(title=f"verify {name}  (accept >= {t.t_accept}, reject < {t.t_reject})")
    for col in ("#", "score", "band", "speech s", "ms"):
        table.add_column(col, justify="right")
    clips = file or [None] * count
    for i, path in enumerate(clips, 1):
        if path is None:
            typer.prompt(f"[{i}/{count}] Press Enter, then speak out loud (3 s)", default="")
            console.print("  🎙  Recording...")
            audio = capture.record(3.0)
        else:
            audio = io.load_audio(path)
        r = v.verify(audio, profile)
        color = {"accept": "green", "uncertain": "yellow", "reject": "red"}[r.band]
        score = "too short" if r.too_short else f"{r.score:.3f}"
        table.add_row(
            str(i), score, f"[{color}]{r.band}[/]", f"{r.speech_seconds:.1f}", f"{r.latency_ms:.0f}"
        )
    console.print(table)


def _print_turn(result) -> None:
    if result is None:
        return
    if result.action is not None:
        a = result.action
        amount = f" ${a.amount}" if a.amount is not None else ""
        console.print(f"  [cyan]action[/] {a.type} {a.counterparty or ''}{amount}".rstrip())
        if result.decision is not None:
            console.print(f"  [cyan]approval[/] {result.decision.outcome}")
    if result.speak is not None:
        s = result.speak
        if s.spoken_text:
            console.print(f"  [green]band says[/] {s.spoken_text}  [dim]({s.channel})[/]")
        if s.phone_text:
            console.print(f"  [magenta]phone[/] {s.phone_text}")


def _trace(events) -> None:
    def show(e) -> None:
        if e.type in ("led", "tts", "spoken", "phone"):
            return
        ms = f" {e.latency_ms:.0f} ms" if e.latency_ms is not None else ""
        data = {k: v for k, v in e.data.items() if k != "result"}
        console.print(f"  [dim]· {e.type}{ms} {data}[/]")

    events.subscribe(show)


@app.command()
def chat(
    speak: bool = typer.Option(False, "--speak", help="Play replies through Piper."),
    phone: bool = typer.Option(False, "--phone", help="Send phone messages to ntfy."),
    trace: bool = typer.Option(False, "--trace", help="Print trace events."),
) -> None:
    """Text mode: type what you'd say to the band."""
    import os

    from earshot.agent.channels import PhoneChannel
    from earshot.agent.events import bus
    from earshot.agent.pipeline import Pipeline
    from earshot.audio.tts import NullTTS, PiperTTS

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pipeline = Pipeline(
        tts=PiperTTS() if speak else NullTTS(), phone=PhoneChannel(console_only=not phone)
    )
    if trace:
        _trace(bus)
    console.print("Type a command (Ctrl-D to quit). Try: order my usual")
    while True:
        try:
            text = console.input("[bold]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if text:
            _print_turn(pipeline.run_text(text))


@app.command()
def live(
    phone: bool = typer.Option(True, "--phone/--no-phone", help="Send phone messages to ntfy."),
    trace: bool = typer.Option(False, "--trace", help="Print trace events."),
) -> None:
    """Mic mode: speak one command at a time; the band answers out loud."""
    import os

    from earshot.agent.channels import PhoneChannel
    from earshot.agent.events import bus
    from earshot.agent.pipeline import LiveMic, Pipeline
    from earshot.audio.capture import MicStream, input_device_name

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    console.print(f"Loading models... (mic: [bold]{input_device_name()}[/])")
    with MicStream() as stream:
        mic = LiveMic(stream)
        pipeline = Pipeline(phone=PhoneChannel(console_only=not phone), listen=mic.listen)
        pipeline.warm()
        if trace:
            _trace(bus)
        console.print("[green]Listening.[/] Say a command. Ctrl-C to quit.")
        try:
            while True:
                segment = mic.next_segment()
                result = pipeline.run_audio(segment)
                if result is not None:
                    console.print(f"[bold]you>[/] {result.text}")
                    _print_turn(result)
        except KeyboardInterrupt:
            console.print("\nStopped.")
