"""Pytest conftest: ensure repo-root packages (`data`, `eval`) are importable."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
