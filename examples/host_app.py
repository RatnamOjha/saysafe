"""A tiny terminal host app showing where Saysafe sits in a voice-agent flow.

Run from the repository root with `uv run python examples/host_app.py`.

The assistant, bank tool, speaker, and phone are simulated. Saysafe's privacy and
approval checks are real. No audio is recorded and no transfer is made.
"""

from __future__ import annotations

import secrets
from decimal import Decimal

from saysafe import Action, ApprovalGuard, PrivacyGuard, Room

ROOMS = {
    "1": (Room.ALONE, "Alone"),
    "2": (Room.UNKNOWN, "Not sure"),
    "3": (Room.OTHERS_PRESENT, "Other people may hear"),
}


def choose_room() -> tuple[Room, str]:
    print("\nWho might hear the assistant? This stands in for your app's room detection.")
    for key, (_, label) in ROOMS.items():
        print(f"  {key}. {label}")
    while True:
        choice = input("Choose 1–3 [2]: ").strip() or "2"
        if choice in ROOMS:
            return ROOMS[choice]
        print("Please choose 1, 2, or 3.")


def show_reply(privacy: PrivacyGuard, text: str, room: Room, sources: set[str]) -> None:
    """The host calls Saysafe before sending assistant text to its speaker."""
    decision = privacy.check(text, sources=sources, room=room)
    print("\nAssistant reply checked by Saysafe before output.")
    print(f"Speaker (simulated): {decision.say}")
    if decision.send_to_phone is not None:
        # A real host sends this only to the authenticated user's private device.
        print(f"Phone (simulated): {decision.send_to_phone}")


def show_phone_step(step) -> None:
    """Display the details the real phone UI must show before its approval button."""
    print(f"\nPhone request (simulated): {step.phone_request}")
    print(f"Full details to show before approval: {step.phone_details}")
    print("No approval is simulated: this example has no authenticated phone channel.")


def fake_bank_transfer(action: Action) -> str:
    """Stand-in for the host app's real external tool call."""
    amount = action.amount or Decimal("0")
    return (
        f"Transfer complete. You sent ${amount:.2f} to {action.counterparty}. "
        "Your checking balance is $480."
    )


def transfer_money(approvals: ApprovalGuard, privacy: PrivacyGuard, room: Room) -> None:
    # The fake assistant proposes one known-contact transfer. A real app builds this
    # Action from the tool call it is considering, then keeps it unchanged through review.
    action = Action(
        type="send_money",
        counterparty="Maya",
        amount=Decimal("25"),
        destination="@maya",
        is_new_counterparty=False,
        source="user_voice",
    )

    step = approvals.start(action, room=room)
    print("\nAssistant's proposed action: send $25 to Maya")
    if step.status == "awaiting_phone":
        # The policy requires private review when the read-back may be overheard.
        print(f"Speaker (simulated): {step.say}")
        show_phone_step(step)
        return
    if step.status != "awaiting_reply":
        print(f"No transfer: approval status is {step.status}.")
        return

    print(f"Speaker (simulated read-back): {step.say}")
    transcript = input("Type the one-time word you heard (blank cancels): ").strip()
    if not transcript:
        step = approvals.reply(step, None)
    else:
        # This fixed passing score is a stand-in for the host's voice matcher.
        # Typing a word does not prove identity; a real app must pass its own score.
        step = approvals.reply(step, transcript, voice_score=0.99)

    if step.status == "awaiting_phone":
        print(f"Speaker (simulated): {step.say}")
        show_phone_step(step)
        return
    if step.status != "approved" or step.token is None:
        if step.say:
            print(f"Speaker (simulated): {step.say}")
        print(f"No transfer: approval status is {step.status}.")
        return

    # Keep verify immediately beside the external action. If any Action field changed,
    # verify raises Refused and the executor must not run.
    approvals.verify(step.token, action)
    result = fake_bank_transfer(action)
    print("Executor (simulated): transfer ran only after token verification.")

    # The tool result takes the same private-reply path as any other assistant reply.
    show_reply(privacy, result, room, sources={"bank"})


def main() -> None:
    print("Saysafe example host app")
    print("Input, speaker, phone, assistant, and bank are simulated; Saysafe checks are real.")
    print("Flow: assistant/tool proposal → Saysafe → speaker, private device, or executor")

    # In production, load one stable secret from the host's secret manager. This demo
    # creates a temporary key for this single process.
    approvals = ApprovalGuard(secret=secrets.token_bytes(32))
    privacy = PrivacyGuard()

    while True:
        room, room_label = choose_room()
        print(f"Room state passed to Saysafe: {room_label}")
        print("\nWhat should the example assistant do?")
        print("  1. Return a bank login code (private reply check)")
        print("  2. Send $25 to Maya (approval check + private result check)")
        print("  q. Quit")
        choice = input("Choose: ").strip().lower()

        if choice == "1":
            # Stand-in for an assistant response built from a sensitive tool result.
            show_reply(
                privacy,
                "Your bank login code is 482913.",
                room,
                sources={"otp"},
            )
        elif choice == "2":
            transfer_money(approvals, privacy, room)
        elif choice == "q":
            print("Goodbye.")
            return
        else:
            print("Please choose 1, 2, or q.")


if __name__ == "__main__":
    main()
