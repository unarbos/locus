"""Task plugin registry for Locus v3."""
from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable


KNOWN_TASKS = {"mlp"}
STREAMING_TASKS = {"gpt_pipe"}


@runtime_checkable
class RoundTask(Protocol):
    N_UB: int
    INNER_REPLICAS: int

    def graph_bundle(self): ...
    def initial_weights(self): ...
    def build_reduce_graph(self, n_inputs: int): ...


@runtime_checkable
class StreamingTask(Protocol):
    def bootstrap(self, *, bucket, run_id: str, max_rounds: int) -> None: ...
    def build_streaming_inputs(self, *, bucket, run_id: str): ...


def load_task(name: str):
    if name in STREAMING_TASKS:
        raise ValueError(
            f"task {name!r} is a streaming task. Its IR builders are available "
            "under locus_tasks.gpt_pipe, but the v3 streaming scheduler is not "
            "wired into RunManager yet; use 'mlp' for the round scheduler."
        )
    if name not in KNOWN_TASKS:
        known = sorted(KNOWN_TASKS | STREAMING_TASKS)
        raise ValueError(f"unknown v3 task {name!r}; known={known}")
    return importlib.import_module(f"locus_tasks.{name}")


def load_streaming_task(name: str):
    if name not in STREAMING_TASKS:
        raise ValueError(f"unknown v3 streaming task {name!r}; known={sorted(STREAMING_TASKS)}")
    return importlib.import_module(f"locus_tasks.{name}")
