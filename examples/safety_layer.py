"""A small, offline example of the Saysafe safety flow.

Run with `uv run python examples/safety_layer.py`. It simulates a voice confirmation,
executes the exact approved transfer, then routes the result privately in a shared room.
No microphone, phone service, LLM, or real money is used.
"""

import secrets

from saysafe import Action, ApprovalGuard, PrivacyGuard, Room


def run_demo() -> tuple[str, str, str | None]:
    # In a real app, keep this fixed server-side. This example uses a throwaway key.
    approvals = ApprovalGuard(secret=secrets.token_bytes(32))
    action = Action(
        type="send_money",
        counterparty="Maya",
        amount=25,
        destination="@maya",
    )

    # 1. Saysafe chooses the approval steps and reads back this exact transfer.
    step = approvals.start(action, room=Room.ALONE)
    if step.status != "awaiting_reply":
        raise RuntimeError(f"Expected a voice challenge, got {step.status!r}")
    spoken_confirmation = step.say

    # 2. Stand in for STT and voice ID: the owner says the one-time word, and
    #    the app supplies a passing voice score.
    step = approvals.reply(step, step.challenge_word, voice_score=0.99)
    if step.status != "approved" or step.token is None:
        raise RuntimeError(f"Transfer was not approved: {step.status!r}")

    # 3. The executor checks the token against the exact action immediately before use.
    approvals.verify(step.token, action)
    result = "Transfer complete. You sent $25 to Maya. Your checking balance is $480."

    # 4. A shared room is reported by the host app; Saysafe returns speech and
    #    private phone text separately.
    decision = PrivacyGuard().check(
        result,
        sources={"bank"},
        room=Room.OTHERS_PRESENT,
    )
    return spoken_confirmation, decision.say, decision.send_to_phone


def main() -> None:
    confirmation, spoken_result, private_result = run_demo()
    print(f"Agent asks: {confirmation}")
    print("Approval: exact transfer approved and verified")
    print(f"Speaker: {spoken_result}")
    if private_result:
        print(f"Phone: {private_result}")


if __name__ == "__main__":
    main()
