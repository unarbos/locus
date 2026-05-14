"""Compatibility helpers for reusing the proven Locus v2 tensor stack.

The v3 subnet/control-plane code is new, but the tensor IR and evaluator are
intentionally reused so every v2 op remains available immediately.
"""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_v2_path() -> None:
    root = Path(__file__).resolve().parents[2]
    v2 = root / "locus_v2"
    if not v2.exists():
        raise ImportError(
            "locus_v3 currently reuses the proven locus_v2 tensor IR/evaluator. "
            f"Expected sibling checkout at {v2}. Vendor those modules or install "
            "from the monorepo layout before running v3."
        )
    if str(v2) not in sys.path:
        sys.path.insert(0, str(v2))
