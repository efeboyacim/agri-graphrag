"""Central configuration, loaded from environment variables / .env."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
# Real environment variables win over .env (docker-compose relies on this for NEO4J_URI).
load_dotenv(ROOT_DIR / ".env", override=False)
# Keep script output readable: no Hugging Face download/progress chatter.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SMALL_MODEL = os.getenv("SMALL_MODEL", "openai/gpt-oss-20b")
LARGE_MODEL = os.getenv("LARGE_MODEL", "openai/gpt-oss-120b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "agrigraph-demo-pw")

_lancedb_path = Path(os.getenv("LANCEDB_PATH", "./data/lancedb"))
LANCEDB_PATH = str(_lancedb_path if _lancedb_path.is_absolute() else ROOT_DIR / _lancedb_path)
LANCEDB_TABLE = "agronomy_notes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def configure_console() -> None:
    """Force UTF-8 stdout/stderr so 'İzmir' doesn't crash printing on a cp1252 Windows pipe."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
