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
    if result is None or result.action is None:
        return
    a = result.action
    amount = f" ${a.amount}" if a.amount is not None else ""
    console.print(f"  [dim]action {a.type} {a.counterparty or ''}{amount}[/]".rstrip())
    if result.decision is not None:
        d = result.decision
        color = {"approve": "green", "step_up": "yellow", "reject": "red"}[d.outcome]
        why = f"  ({'; '.join(d.reasons)})" if d.reasons else ""
        console.print(f"  [{color}]{d.outcome}[/]{why}")


def _narrate(events, show_you: bool = False) -> None:
    """Print what the band says and what reaches the phone, as it happens.
    show_you: also echo the user's transcript (live mode, where nothing was typed)."""

    def show(e) -> None:
        if e.type == "transcript" and e.data.get("text") and show_you:
            console.print(f"[bold]you>[/] {e.data['text']}")
        elif e.type == "spoken":
            where = "" if e.data["channel"] == "speaker" else f" [dim]({e.data['channel']})[/]"
            console.print(f"  [green]band:[/] {e.data['text']}{where}")
        elif e.type == "phone":
            console.print(f"  [magenta]phone ({e.data['via']}):[/] {e.data['text']}")
        elif e.type == "reply_captured":
            if e.data.get("timed_out"):
                console.print("  [dim]heard nothing (timed out)[/]")
            else:
                secs = e.data.get("speech_seconds")
                length = f" ({secs:.1f} s)" if secs is not None else ""
                console.print(f"  [dim]heard {e.data['text']!r}{length}[/]")
        elif e.type == "speaker_scored":
            parts = [f"{k} {e.data[k]:.2f}" for k in ("reply", "command", "fused")
                     if e.data[k] is not None]  # fmt: skip
            score = ", ".join(parts) if parts else "no voice score"
            console.print(f"  [dim]voice: {score} -> {e.data['band']}[/]")

    events.subscribe(show)


def _trace(events) -> None:
    def show(e) -> None:
        if e.type in ("led", "tts", "spoken", "phone", "speaker_scored", "reply_captured"):
            return
        ms = f" {e.latency_ms:.0f} ms" if e.latency_ms is not None else ""
        data = {k: v for k, v in e.data.items() if k != "result"}
        console.print(f"  [dim]· {e.type}{ms} {data}[/]")

    events.subscribe(show)


def _warn_if_not_enrolled(mic: str) -> None:
    from earshot.approvals.hook import get_approver

    scorer = get_approver().scorer
    if scorer is None:
        console.print(
            "[yellow]No owner voice profile loaded (EARSHOT_OWNER in .env).[/] "
            "Every voice approval will step up to a phone tap."
        )
    elif scorer.profile.mic_name != mic:
        console.print(
            f"[yellow]You enrolled with '{scorer.profile.mic_name}' but this is '{mic}'.[/] "
            "Voice scores will be lower. Re-enroll with the mic you'll demo with."
        )


def _watch_audience(tracker) -> None:
    """Print the audience level whenever it changes (it also changes as time passes)."""
    import threading
    import time

    def loop() -> None:
        last = None
        while True:
            state = tracker.state()
            if state.level != last:
                color = {"alone_likely": "green", "unknown": "yellow", "others_present": "red"}
                console.print(
                    f"  [{color[state.level]}]audience: {state.level}[/] "
                    f"[dim]({'; '.join(state.evidence)})[/]"
                )
                last = state.level
            time.sleep(1)

    threading.Thread(target=loop, daemon=True, name="audience-watch").start()


def _typed_reply(timeout_s: float) -> str | None:
    """Chat mode: the approval reply is typed. Typed text carries no voice evidence."""
    try:
        text = console.input("[bold]reply>[/] ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return text or None


@app.command()
def chat(
    speak: bool = typer.Option(False, "--speak", help="Play replies through Piper."),
    phone: bool = typer.Option(False, "--phone", help="Send phone messages to ntfy."),
    trace: bool = typer.Option(False, "--trace", help="Print trace events."),
    room: str = typer.Option(
        "unknown", help="Pretend room: alone, unknown or others (text mode has no mic)."
    ),
    headphones: bool = typer.Option(False, "--headphones", help="Pretend headphones are in."),
    discreet: bool = typer.Option(False, "--discreet", help="Turn on discreet mode."),
) -> None:
    """Text mode: type what you'd say to the band."""
    import os

    from earshot.agent.channels import PhoneChannel
    from earshot.agent.events import bus
    from earshot.agent.pipeline import Pipeline
    from earshot.audio.tts import NullTTS, PiperTTS
    from earshot.privacy.audience import FixedAudience

    if room not in ("alone", "unknown", "others"):
        _fail("--room must be alone, unknown or others")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pipeline = Pipeline(
        tts=PiperTTS() if speak else NullTTS(),
        phone=PhoneChannel(console_only=not phone),
        listen=_typed_reply,
        audience=FixedAudience(
            {"alone": "alone_likely", "unknown": "unknown", "others": "others_present"}[room]
        ),
    )
    pipeline.flags.update(headphones=headphones, discreet_mode=discreet)
    _narrate(bus)
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
    port: int = typer.Option(8000, help="Port for the phone approval routes."),
    audience: bool = typer.Option(False, "--audience", help="Print who's-listening changes."),
    trace: bool = typer.Option(False, "--trace", help="Print trace events."),
) -> None:
    """Mic mode: speak one command at a time; the band answers out loud."""
    import os

    from earshot.agent.channels import PhoneChannel
    from earshot.agent.events import bus
    from earshot.agent.pipeline import LiveMic, Pipeline
    from earshot.audio.capture import MicStream, input_device_name
    from earshot.config import env
    from earshot.server.app import serve_in_background

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    console.print(f"Loading models... (mic: [bold]{input_device_name()}[/])")
    from earshot.privacy.audience import AudienceTracker, owner_score_fn

    tracker = AudienceTracker(score=owner_score_fn())

    def observe(segment) -> None:
        obs = tracker.observe_segment(segment)
        if audience and obs is not None:
            who = {"owner": "you", "other": "other", "unclear": "unclear"}[obs.label]
            score = "?" if obs.score is None else f"{obs.score:.2f}"
            console.print(f"  [dim](voice: {who} {score})[/]")

    with MicStream() as stream:
        mic = LiveMic(stream, on_segment=observe)
        tracker.mic_on()
        pipeline = Pipeline(
            phone=PhoneChannel(console_only=not phone), listen=mic.listen, audience=tracker
        )
        pipeline.warm()
        _warn_if_not_enrolled(input_device_name())
        serve_in_background(port=port)
        console.print(f"Phone approvals: {env('EARSHOT_PUBLIC_URL', 'http://localhost:8000')}")
        _narrate(bus, show_you=True)
        if audience:
            _watch_audience(tracker)
        if trace:
            _trace(bus)
        console.print("[green]Listening.[/] Say a command. Ctrl-C to quit.")
        try:
            while True:
                segment = mic.next_segment()
                _print_turn(pipeline.run_audio(segment))
        except KeyboardInterrupt:
            console.print("\nStopped.")


def demo_actions():
    """Every demo action plus the edge cases the policy has to get right."""
    from decimal import Decimal

    from earshot.agent.mock_agent import MockAgent

    agent = MockAgent()
    order = agent.handle("order my usual")
    return [
        ("order my usual", order),
        ("send fifty dollars to Jake", agent.handle("send fifty dollars to Jake")),
        ("send twenty dollars to Priya", agent.handle("send twenty dollars to Priya")),
        ("cancel my Netflix", agent.handle("cancel my Netflix")),
        ("set a reminder to call mom at six", agent.handle("set a reminder to call mom at six")),
        ("big order ($80)", order.model_copy(update={"amount": Decimal("80")})),
        ("send $250 to Jake", agent.handle("send $250 to Jake")),
        (
            "email says: cancel Netflix",
            agent.handle("cancel my Netflix").model_copy(update={"source": "from_content"}),
        ),
    ]


@app.command()
def policy(demo: bool = typer.Option(False, "--demo", help="Show every demo action.")) -> None:
    """Show the tier, rule, reasons and read-back for actions."""
    from rich.table import Table

    from earshot.approvals.challenge import ChallengeIssuer
    from earshot.approvals.policy import assess
    from earshot.approvals.readback import readback

    if not demo:
        _fail("Use --demo.")
    issuer = ChallengeIssuer()
    table = Table(show_lines=True)
    for col in ("said", "tier", "rule", "reasons", "read-back", "s"):
        table.add_column(col)
    for said, action in demo_actions():
        risk = assess(action)
        word = issuer.issue(action.id).word if risk.tier == "voice_challenge" else None
        rb = readback(action, risk, word)
        tier = risk.tier + (f"\n(bumped from {risk.base_tier})" if risk.bumped else "")
        table.add_row(
            said, tier, risk.rule_id, "\n".join(risk.reasons), rb.text, f"{rb.est_seconds:.1f}"
        )
    console.print(table)


@app.command()
def detect(
    text: str,
    tag: list[str] = typer.Option(None, "--tag", "-t", help="Source hint: otp, bank, health..."),
) -> None:
    """Show how sensitive a reply is: level, flagged spans, latency per layer."""
    from rich.text import Text

    from earshot.privacy.detect import detect as run_detect

    d = run_detect(text, set(tag or []))
    color = {"public": "green", "personal": "cyan", "sensitive": "yellow", "secret": "red"}
    console.print(f"level: [bold {color[d.level]}]{d.level}[/]   categories: {d.categories}")
    marked = Text(text)
    for s in d.spans:
        marked.stylize("bold reverse", s.start, s.end)
    console.print(marked)
    for s in d.spans:
        console.print(f"  [{s.start}:{s.end}] {s.category} ({s.source}): {s.text!r}")
    latency = ", ".join(f"{k} {v:.2f} ms" for k, v in d.latency_ms.items())
    console.print(f"[dim]latency: {latency}[/]")
