"""saysafe: privacy and approval guardrails for voice agents.

Who's listening: PrivacyGuard keeps sensitive replies off the speaker when others may hear.
Who's speaking: ApprovalGuard scales proof to risk and binds approvals to the exact action.
Voice ID (optional): saysafe.voice, with pip install "saysafe[voice]".
Pipecat: saysafe.integrations.pipecat, with pip install "saysafe[pipecat]".
"""

from saysafe.approvals.actions import Action, action_hash
from saysafe.approvals.audit import AuditLog
from saysafe.approvals.fusion import Thresholds, load_thresholds
from saysafe.approvals.guard import ApprovalGuard, Step, StepClosed
from saysafe.approvals.tokens import MemoryNonceStore, Refused, SQLiteNonceStore
from saysafe.privacy.audience import Room
from saysafe.privacy.guard import PrivacyDecision, PrivacyGuard

__all__ = [
    "Action",
    "ApprovalGuard",
    "AuditLog",
    "MemoryNonceStore",
    "PrivacyDecision",
    "PrivacyGuard",
    "Refused",
    "Room",
    "SQLiteNonceStore",
    "Step",
    "StepClosed",
    "Thresholds",
    "__version__",
    "action_hash",
    "load_thresholds",
]

__version__ = "0.1.0"
