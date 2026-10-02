"""Band simulator demo: a screenless voice assistant with saysafe in front of it."""

from pathlib import Path

from dotenv import load_dotenv

# The repo's .env configures the demo. Real environment wins, so tests can override anything.
load_dotenv(Path(__file__).resolve().parents[4] / ".env", override=False)
