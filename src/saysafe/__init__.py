"""saysafe: privacy and approval guardrails for voice agents.

Who's listening: PrivacyGuard keeps sensitive replies off the speaker when others may hear.
Who's speaking: saysafe.approvals scales proof to risk and binds approvals to the exact action.
Voice ID (optional): saysafe.voice, with pip install "saysafe[voice]".
Pipecat: saysafe.integrations.pipecat, with pip install "saysafe[pipecat]".
"""

from saysafe.privacy.audience import Room
from saysafe.privacy.guard import PrivacyDecision, PrivacyGuard

__all__ = ["PrivacyDecision", "PrivacyGuard", "Room", "__version__"]

__version__ = "0.1.0"
