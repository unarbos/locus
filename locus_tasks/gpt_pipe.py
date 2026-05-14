"""V3 compatibility surface for the v2 GPT pipeline task.

The full staged GPT IR is reused from v2 for now. The v3 orchestrator vertical
slice uses `mlp`; this module keeps the GPT graph builders available for the
next streaming scheduler adapter without losing any implemented ops.
"""
from __future__ import annotations

from locus_core.compat import ensure_v2_path

ensure_v2_path()

from locus.tasks.gpt_pipe import *  # noqa: F401,F403
