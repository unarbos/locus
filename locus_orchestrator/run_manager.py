"""Run manager and job emitter for Locus v3."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from locus_core import paths
from locus_core.protocol import ArtifactRef, GraphRef, JobManifestV3, MinerIdentity, VerificationPolicy, WorkerIdentity
from locus_runtime import tensor_io
from locus_runtime.storage import ObjectStore
from locus_tasks import load_task
from .scheduler import CriticalGate, QuotaBook


@dataclass
class RunConfig:
    netuid: int
    run_id: str
    task: str = "mlp"
    max_steps: int = 1
    owner_secret: str = "owner-dev-secret"


class RunManager:
    def __init__(self, *, bucket: ObjectStore, config: RunConfig) -> None:
        self.bucket = bucket
        self.config = config
        self.task = load_task(config.task)
        self.graphs = self.task.graph_bundle()
        self.quota = QuotaBook()
        self.gate = CriticalGate()
        self.emitted: list[str] = []

    def bootstrap(self) -> None:
        w0, w1 = self.task.initial_weights()
        self.bucket.put(
            self.bucket.uri_for_key(paths.weights_key(self.config.netuid, self.config.run_id, 0, 0)),
            tensor_io.encode_tensor(w0),
        )
        self.bucket.put(
            self.bucket.uri_for_key(paths.weights_key(self.config.netuid, self.config.run_id, 0, 1)),
            tensor_io.encode_tensor(w1),
        )
        for graph in self.graphs.values():
            sha = graph.graph_id()
            self.bucket.put(
                self.bucket.uri_for_key(paths.graph_key(self.config.netuid, self.config.run_id, sha)),
                graph.to_canonical_json(),
            )
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.manifest_config_key(self.config.netuid, self.config.run_id)),
            {
                "task": self.config.task,
                "max_steps": self.config.max_steps,
                "netuid": self.config.netuid,
            },
        )
        self._save_state({"run_id": self.config.run_id, "current_step": 0, "max_steps": self.config.max_steps})

    def discover_workers(self) -> list[WorkerIdentity]:
        prefix = self.bucket.uri_for_key(paths.miners_prefix(self.config.netuid))
        out: list[WorkerIdentity] = []
        for uri in self.bucket.list(prefix):
            if not uri.endswith("/heartbeat.json"):
                continue
            try:
                data = self.bucket.get_json(uri)
                if data.get("run_id") != self.config.run_id:
                    continue
                out.append(WorkerIdentity.from_dict(data["worker"]))
            except Exception:
                continue
        identities = [MinerIdentity(netuid=self.config.netuid, hotkey_ss58=w.hotkey_ss58) for w in out]
        self.quota.update_workers(identities, out)
        return out

    def run_loop(self, *, poll_interval: float = 0.05, timeout_sec: float = 60.0) -> None:
        if not self.bucket.exists(self.bucket.uri_for_key(paths.state_key(self.config.netuid, self.config.run_id))):
            self.bootstrap()
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self.wait_for_workers(deadline=deadline, poll_interval=poll_interval)
            state = self.bucket.get_json(self.bucket.uri_for_key(paths.state_key(self.config.netuid, self.config.run_id)))
            step = int(state.get("current_step", 0))
            if step >= self.config.max_steps:
                return
            self.run_step(step)
            self._save_state({"run_id": self.config.run_id, "current_step": step + 1, "max_steps": self.config.max_steps})
            time.sleep(poll_interval)
        raise TimeoutError(f"orchestrator timed out for run {self.config.run_id}")

    def wait_for_workers(self, *, deadline: float, poll_interval: float) -> None:
        while time.time() < deadline:
            workers = self.discover_workers()
            if workers:
                return
            self._save_state({
                "run_id": self.config.run_id,
                "current_step": self.bucket.get_json(
                    self.bucket.uri_for_key(paths.state_key(self.config.netuid, self.config.run_id))
                ).get("current_step", 0),
                "max_steps": self.config.max_steps,
                "status": "waiting_for_miners",
            })
            time.sleep(poll_interval)
        raise TimeoutError(f"no miners heartbeated for run {self.config.run_id}")

    def run_step(self, step: int) -> None:
        fwd = self.emit_forward(step)
        self.wait_outputs(fwd)
        inners = [self.emit_inner(step, ub, replica) for ub in range(self.task.N_UB) for replica in range(self.task.INNER_REPLICAS)]
        for job in inners:
            self.wait_outputs(job)
        reduces = [self.emit_reduce(step, ub) for ub in range(self.task.N_UB)]
        for job in reduces:
            self.wait_outputs(job)
        outers = [self.emit_outer(step, ub) for ub in range(self.task.N_UB)]
        for job in outers:
            self.wait_outputs(job)
        eval_job = self.emit_eval(step)
        self.wait_outputs(eval_job)

    def emit_forward(self, step: int) -> JobManifestV3:
        worker = self.quota.pick_worker()
        outputs = [
            ArtifactRef(name=f"target_{ub}", uri=self.bucket.uri_for_key(paths.target_key(self.config.netuid, self.config.run_id, step, ub)))
            for ub in range(self.task.N_UB)
        ]
        inputs = [
            ArtifactRef(name=f"weights_{ub}", uri=self.bucket.uri_for_key(paths.weights_key(self.config.netuid, self.config.run_id, step, ub)))
            for ub in range(self.task.N_UB)
        ]
        return self.emit_job("forward_pass", step, self.graphs["forward"], {"round_id": step}, inputs, outputs, worker)

    def emit_inner(self, step: int, ub: int, replica: int) -> JobManifestV3:
        worker = self.quota.pick_worker()
        job_id = f"step{step}-ub{ub}-inner-r{replica}"
        outputs = [
            ArtifactRef(
                name="delta",
                uri=self.bucket.uri_for_key(
                    paths.artifact_key(self.config.netuid, self.config.run_id, job_id, worker.hotkey_ss58, worker.worker_id, 0, "delta")
                ),
            )
        ]
        inputs = [
            ArtifactRef(name="weights", uri=self.bucket.uri_for_key(paths.weights_key(self.config.netuid, self.config.run_id, step, ub))),
            ArtifactRef(name="target", uri=self.bucket.uri_for_key(paths.target_key(self.config.netuid, self.config.run_id, step, ub))),
        ]
        return self.emit_job("inner_step", step, self.graphs["inner"], {"ub": ub, "replica": replica}, inputs, outputs, worker, job_id=job_id)

    def emit_reduce(self, step: int, ub: int) -> JobManifestV3:
        worker = self.quota.pick_worker()
        deltas = []
        for jid in self.emitted:
            if f"step{step}-ub{ub}-inner" in jid:
                manifest = self.load_job(jid)
                deltas.extend(manifest.outputs)
        graph = self.task.build_reduce_graph(len(deltas))
        outputs = [
            ArtifactRef(
                name="reduced",
                uri=self.bucket.uri_for_key(paths.artifact_key(self.config.netuid, self.config.run_id, f"step{step}-ub{ub}-reduce", worker.hotkey_ss58, worker.worker_id, 0, "reduced")),
            )
        ]
        inputs = [ArtifactRef(name=f"d_{i}", uri=ref.uri) for i, ref in enumerate(deltas)]
        return self.emit_job("reduce", step, graph, {"ub": ub, "n_inputs": len(inputs)}, inputs, outputs, worker, job_id=f"step{step}-ub{ub}-reduce")

    def emit_outer(self, step: int, ub: int) -> JobManifestV3:
        worker = self.quota.pick_worker()
        reduce_job = self.load_job(f"step{step}-ub{ub}-reduce")
        inputs = [
            ArtifactRef(name="weights", uri=self.bucket.uri_for_key(paths.weights_key(self.config.netuid, self.config.run_id, step, ub))),
            ArtifactRef(name="reduced_delta", uri=reduce_job.outputs[0].uri),
        ]
        outputs = [
            ArtifactRef(name="new_weights", uri=self.bucket.uri_for_key(paths.weights_key(self.config.netuid, self.config.run_id, step + 1, ub)))
        ]
        return self.emit_job("outer_step", step, self.graphs["outer"], {"ub": ub}, inputs, outputs, worker, job_id=f"step{step}-ub{ub}-outer")

    def emit_eval(self, step: int) -> JobManifestV3:
        worker = self.quota.pick_worker()
        inputs = [
            ArtifactRef(name=f"weights_{ub}", uri=self.bucket.uri_for_key(paths.weights_key(self.config.netuid, self.config.run_id, step + 1, ub)))
            for ub in range(self.task.N_UB)
        ]
        outputs = [
            ArtifactRef(name="metrics", uri=self.bucket.uri_for_key(f"{paths.run_root(self.config.netuid, self.config.run_id)}/metrics/step={step}.json"))
        ]
        return self.emit_job("eval", step, self.graphs["eval"], {"round_id": step + 1}, inputs, outputs, worker, job_id=f"step{step}-eval")

    def emit_job(
        self,
        kind: str,
        step: int,
        graph,
        params: dict,
        inputs: list[ArtifactRef],
        outputs: list[ArtifactRef],
        worker: WorkerIdentity,
        *,
        job_id: str | None = None,
    ) -> JobManifestV3:
        now = int(time.time())
        sha = graph.graph_id()
        graph_uri = self.bucket.uri_for_key(paths.graph_key(self.config.netuid, self.config.run_id, sha))
        if not self.bucket.exists(graph_uri):
            self.bucket.put(graph_uri, graph.to_canonical_json())
        job_id = job_id or f"step{step}-{kind}"
        manifest = JobManifestV3(
            job_id=job_id,
            run_id=self.config.run_id,
            step_id=step,
            kind=kind,
            graph_ref=GraphRef(sha256=sha, uri=graph_uri),
            params=params,
            inputs=inputs,
            outputs=outputs,
            assigned_hotkey=worker.hotkey_ss58,
            assigned_worker=worker.worker_id,
            attempt=0,
            deadline_unix=now + 600,
            created_unix=now,
            verification_policy=VerificationPolicy(critical=kind in {"outer_step"}),
        ).sign(self.config.owner_secret)
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, self.config.run_id, job_id)),
            manifest.to_dict(),
        )
        self.emitted.append(job_id)
        self.bucket.put_json(
            self.bucket.uri_for_key(paths.job_index_key(self.config.netuid, self.config.run_id)),
            self.emitted,
        )
        step_index_uri = self.bucket.uri_for_key(
            paths.job_step_index_key(self.config.netuid, self.config.run_id, step)
        )
        try:
            step_jobs = self.bucket.get_json(step_index_uri) if self.bucket.exists(step_index_uri) else []
        except Exception:
            step_jobs = []
        if job_id not in step_jobs:
            step_jobs.append(job_id)
        self.bucket.put_json(step_index_uri, step_jobs)
        return manifest

    def load_job(self, job_id: str) -> JobManifestV3:
        return JobManifestV3.from_dict(
            self.bucket.get_json(self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, self.config.run_id, job_id)))
        )

    def wait_outputs(self, manifest: JobManifestV3, *, timeout_sec: float = 30.0) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if all(self.bucket.exists(ref.uri) for ref in manifest.outputs):
                self.quota.release(manifest.assigned_hotkey)
                return
            time.sleep(0.05)
        self.quota.release(manifest.assigned_hotkey)
        self.bucket.put_json(
            self.bucket.uri_for_key(
                f"{paths.jobs_prefix(self.config.netuid, self.config.run_id)}{manifest.job_id}/stale.json"
            ),
            {"job_id": manifest.job_id, "stale_unix": int(time.time()), "reason": "output_timeout"},
        )
        raise TimeoutError(f"timed out waiting for {manifest.job_id}")

    def _save_state(self, state: dict) -> None:
        self.bucket.put_json(self.bucket.uri_for_key(paths.state_key(self.config.netuid, self.config.run_id)), state)
