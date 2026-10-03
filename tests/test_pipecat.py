import re

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.tests.utils import run_test  # noqa: E402

from saysafe import Room  # noqa: E402
from saysafe.integrations.pipecat import PHONE_FAILED_LINE, PrivacyFilter  # noqa: E402
from saysafe.privacy.route import PHONE_ONLY_LINE  # noqa: E402

pytestmark = pytest.mark.asyncio


class Phone:
    def __init__(self, fail: bool = False):
        self.sent: list[str] = []
        self.fail = fail

    def __call__(self, text: str) -> None:
        if self.fail:
            raise ConnectionError("no network")
        self.sent.append(text)


def reply(text: str) -> list:
    """An LLM reply streamed the way LLM services do: word-sized chunks with spaces."""
    chunks = re.findall(r"\S+\s*", text)
    return [LLMFullResponseStartFrame(), *[LLMTextFrame(text=c) for c in chunks],
            LLMFullResponseEndFrame()]  # fmt: skip


async def run(processor, frames):
    down, _ = await run_test(processor, frames_to_send=frames)
    return down


def spoken(down) -> str:
    return "".join(f.text for f in down if isinstance(f, LLMTextFrame)).strip()


async def test_public_reply_streams_through_sentence_by_sentence():
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone, room=Room.OTHERS_PRESENT)
    down = await run(f, reply("It will be sunny today. Bring sunglasses!"))
    texts = [x.text for x in down if isinstance(x, LLMTextFrame)]
    assert texts == ["It will be sunny today.", " Bring sunglasses!"]
    assert isinstance(down[0], LLMFullResponseStartFrame)
    assert isinstance(down[-1], LLMFullResponseEndFrame)
    assert phone.sent == []


async def test_code_goes_to_the_phone_when_others_may_hear():
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone, room=Room.OTHERS_PRESENT)
    down = await run(f, reply("Your Chase code is 482913."))
    assert spoken(down) == PHONE_ONLY_LINE
    assert phone.sent == ["Your Chase code is 482913."]


async def test_number_explained_by_an_earlier_sentence_is_caught():
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone, room=Room.OTHERS_PRESENT)
    down = await run(f, reply("Your Chase code is ready. It's 482913."))
    assert spoken(down) == f"Your Chase code is ready. {PHONE_ONLY_LINE}"
    assert "482913" not in spoken(down)
    assert phone.sent == ["Your Chase code is ready. It's 482913."]


async def test_number_explained_by_a_later_sentence_is_caught():
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone, room=Room.OTHERS_PRESENT)
    down = await run(f, reply("Sure. It's 482913. That's your Chase code."))
    assert "482913" not in spoken(down)
    assert spoken(down).startswith("Sure.")
    assert phone.sent == ["Sure. It's 482913. That's your Chase code."]


async def test_redacted_reply_keeps_the_useful_part():
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone)  # room unknown
    down = await run(f, reply("Your rent of $2,400 is due Friday."))
    assert spoken(down) == "Your rent is due Friday. Details are on your phone."
    assert phone.sent == ["Your rent of $2,400 is due Friday."]


async def test_alone_or_in_headphones_everything_is_said():
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone, room=Room.ALONE)
    assert spoken(await run(f, reply("Your Chase code is 482913."))) == (
        "Your Chase code is 482913."
    )
    f = PrivacyFilter(send_to_phone=phone, room=Room.OTHERS_PRESENT)
    f.headphones = True
    assert spoken(await run(f, reply("Your Chase code is 482913."))) == (
        "Your Chase code is 482913."
    )
    assert phone.sent == []


async def test_room_can_be_read_from_device_state():
    device = {"room": Room.ALONE}
    f = PrivacyFilter(send_to_phone=Phone(), room=lambda: device["room"])
    device["room"] = Room.OTHERS_PRESENT
    assert spoken(await run(f, reply("Your Chase code is 482913."))) == PHONE_ONLY_LINE


async def test_tool_results_tag_the_next_reply_only():
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone, room=Room.OTHERS_PRESENT,
                      tool_sources={"get_login_code": {"otp"}})  # fmt: skip
    result = FunctionCallResultFrame(
        function_name="get_login_code", tool_call_id="1", arguments={}, result="482913"
    )
    frames = [result, *reply("Here it is: four eight two nine one three."),
              *reply("Anything else?")]  # fmt: skip
    down = await run(f, frames)
    assert spoken(down) == f"{PHONE_ONLY_LINE}Anything else?"
    assert phone.sent == ["Here it is: four eight two nine one three."]


async def test_failed_delivery_never_says_the_secret():
    f = PrivacyFilter(send_to_phone=Phone(fail=True), room=Room.OTHERS_PRESENT)
    down = await run(f, reply("Your Chase code is 482913."))
    assert spoken(down) == PHONE_FAILED_LINE


async def test_async_phone_sender():
    sent = []

    async def push(text):
        sent.append(text)

    f = PrivacyFilter(send_to_phone=push, room=Room.OTHERS_PRESENT)
    await run(f, reply("Your Chase code is 482913."))
    assert sent == ["Your Chase code is 482913."]


async def test_reply_without_final_punctuation_is_still_checked():
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone, room=Room.OTHERS_PRESENT)
    down = await run(f, reply("Your Chase code is 482913"))
    assert spoken(down) == PHONE_ONLY_LINE


async def test_tts_speak_frames_are_checked():
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone, room=Room.OTHERS_PRESENT)
    down = await run(f, [TTSSpeakFrame(text="Welcome back!"),
                         TTSSpeakFrame(text="Your Chase code is 482913.")])  # fmt: skip
    said = [x.text for x in down if isinstance(x, TTSSpeakFrame)]
    assert said == ["Welcome back!", PHONE_ONLY_LINE]
    assert phone.sent == ["Your Chase code is 482913."]


async def test_interrupted_reply_drops_what_was_held():
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone, room=Room.OTHERS_PRESENT)
    frames = [*reply("Your Chase code is 482913.")[:-1], InterruptionFrame(),
              *reply("Okay.")]  # fmt: skip
    down = await run(f, frames)
    assert "482913" not in spoken(down)
    assert phone.sent == []


async def test_decision_event():
    decisions = []
    f = PrivacyFilter(send_to_phone=Phone(), room=Room.OTHERS_PRESENT)

    @f.event_handler("on_decision")
    async def on_decision(processor, decision):
        decisions.append(decision)

    await run(f, reply("Your Chase code is 482913."))
    assert [d.channel for d in decisions] == ["phone_only"]


@pytest.mark.parametrize("size", [1, 3, 7])
@pytest.mark.parametrize(
    "text",
    ["Your Chase code is 482913.", "Sure. It's 482913. That's your Chase code.",
     "Your Chase code is ready. It's 4 8 2 9 1 3."],
)  # fmt: skip
async def test_any_chunking_gives_the_same_result(size, text):
    phone = Phone()
    f = PrivacyFilter(send_to_phone=phone, room=Room.OTHERS_PRESENT)
    chunks = [text[i : i + size] for i in range(0, len(text), size)]
    frames = [LLMFullResponseStartFrame(), *[LLMTextFrame(text=c) for c in chunks],
              LLMFullResponseEndFrame()]  # fmt: skip
    down = await run(f, frames)
    assert not re.search(r"4\s?8\s?2\s?9\s?1\s?3", spoken(down))
    assert phone.sent == [text]
