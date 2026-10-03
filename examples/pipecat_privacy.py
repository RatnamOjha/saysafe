"""saysafe's PrivacyFilter in a real Pipecat pipeline, text only (no mic, no TTS).

    pip install "saysafe[pipecat]"
    GROQ_API_KEY=... python examples/pipecat_privacy.py

Asks the same questions with nobody else around and with others in the room, and
prints what the speaker would say and what goes to the phone. The tools return fake data.
"""

import asyncio
import os

from dotenv import load_dotenv
from pipecat.frames.frames import (
    EndFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.workers.runner import WorkerRunner

from saysafe import Room
from saysafe.integrations.pipecat import PrivacyFilter

QUESTIONS = [
    "What's my Chase login code?",
    "How much is in my checking account?",
    "How far is the Moon?",
]


async def get_login_code(params: FunctionCallParams, bank: str):
    """Get the user's latest one-time login code from their bank.

    Args:
        bank: The bank's name.
    """
    await params.result_callback({"bank": bank, "code": "482913"})


async def get_balance(params: FunctionCallParams):
    """Get the balance of the user's checking account."""
    await params.result_callback({"balance_usd": 2412.55})


class Speaker(FrameProcessor):
    """Stands in for TTS: collects what would be said, one reply at a time."""

    def __init__(self):
        super().__init__()
        self.text = ""
        self.done = asyncio.Event()

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self.text += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame) and self.text.strip():
            print(f"  speaker: {self.text.strip()}")
            self.text = ""
            self.done.set()
        await self.push_frame(frame, direction)


async def conversation(room: Room, api_key: str) -> None:
    print(f"\nRoom: {room.name}")
    context = LLMContext(tools=[get_login_code, get_balance])
    aggregators = LLMContextAggregatorPair(context)
    llm = GroqLLMService(api_key=api_key, settings=GroqLLMService.Settings(system_instruction=(
        "You are a voice assistant on a wearable. Answer in one or two short sentences, "
        "and use the tools for the user's accounts.")))  # fmt: skip

    async def send_to_phone(text: str) -> None:
        print(f"  phone:   {text}")

    privacy = PrivacyFilter(
        send_to_phone=send_to_phone,
        room=room,
        tool_sources={"get_login_code": {"otp"}, "get_balance": {"bank"}},
    )
    speaker = Speaker()
    worker = PipelineWorker(
        Pipeline([aggregators.user(), llm, privacy, speaker, aggregators.assistant()]),
        cancel_on_idle_timeout=False,
    )
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_started(worker, frame):
        started.set()

    async def ask_all():
        await started.wait()
        for question in QUESTIONS:
            print(f"  user:    {question}")
            speaker.done.clear()
            message = {"role": "user", "content": question}
            await worker.queue_frame(LLMMessagesAppendFrame(messages=[message], run_llm=True))
            await asyncio.wait_for(speaker.done.wait(), timeout=60)
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    await asyncio.gather(runner.run(), ask_all())


async def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY (free at console.groq.com).")
    for room in (Room.ALONE, Room.OTHERS_PRESENT):
        await conversation(room, api_key)


if __name__ == "__main__":
    asyncio.run(main())
