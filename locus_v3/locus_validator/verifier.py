"""Replay verification and score windows for Locus v3."""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any

import torch

from locus_core import paths
from locus_core.ir import Graph
from locus_core.protocol import ArtifactDigest, ArtifactRef, JobManifestV3, JobReceiptV3, MinerScoreWindow, VerificationVerdictV3
from locus_core.signatures import HmacSigner, verify_dict
from locus_runtime import tensor_io
from locus_runtime.crypto import decode_envelope, TimelockPending, DrandTimelockProvider
from locus_runtime.eval import evaluate
from locus_runtime.storage import ObjectStore


@dataclass
class ValidatorConfig:
    netuid: int
    run_id: str
    validator_hotkey: str
    validator_secret: str = "validator-dev-secret"
    owner_secret: str = "owner-dev-secret"
    miner_secret: str = "miner-dev-secret"
    device: str = "cpu"
    sample_rate: float = 1.0
    max_sample_elements: int = 4096
    encryption_secret: str = "locus-dev-encryption"
    timelock_provider: DrandTimelockProvider | None = None


class ReplayVerifier:
    def __init__(self, *, bucket: ObjectStore, config: ValidatorConfig) -> None:
        self.bucket = bucket
        self.config = config
        self.graph_cache: dict[str, Graph] = {}

    def run_once(self, *, max_receipts: int | None = None) -> int:
        checked = 0
        for uri, receipt in self.sample_receipts():
            if max_receipts is not None and checked >= max_receipts:
                break
            if self.has_verdict(receipt):
                continue
            verdict = self.verify(uri, receipt)
            self.bucket.put_json(
                self.bucket.uri_for_key(paths.verdict_key(self.config.netuid, receipt.run_id, self.config.validator_hotkey, receipt.receipt_id)),
                verdict.to_dict(),
            )
            checked += 1
        return checked

    def sample_receipts(self) -> list[tuple[str, JobReceiptV3]]:
        prefix = self.bucket.uri_for_key(paths.receipts_prefix(self.config.netuid, self.config.run_id))
        out: list[tuple[str, JobReceiptV3]] = []
        for uri in self.bucket.list(prefix):
            if not uri.endswith(".json"):
                continue
            try:
                receipt = JobReceiptV3.from_dict(self.bucket.get_json(uri))
            except Exception:
                continue
            if self.config.sample_rate < 1.0:
                h = hashlib.sha256(f"{self.config.validator_hotkey}:{receipt.receipt_id}".encode()).digest()
                if int.from_bytes(h[:8], "big") / float(2**64) >= self.config.sample_rate:
                    continue
            out.append((uri, receipt))
        random.Random(17).shuffle(out)
        return out

    def has_verdict(self, receipt: JobReceiptV3) -> bool:
        return self.bucket.exists(
            self.bucket.uri_for_key(paths.verdict_key(self.config.netuid, receipt.run_id, self.config.validator_hotkey, receipt.receipt_id))
        )

    def verify(self, receipt_uri: str, receipt: JobReceiptV3) -> VerificationVerdictV3:
        t0 = time.time()
        comparison: dict[str, Any] = {"receipt_uri": receipt_uri, "inputs": {}, "outputs": {}}
        try:
            manifest = self.find_manifest(receipt)
            if not self.verify_manifest_signature(manifest):
                return self.verdict(receipt, "fail", "bad owner signature", 0.0, comparison, t0)
            if not self.verify_receipt_signature(receipt):
                return self.verdict(receipt, "fail", "bad miner signature", 0.0, comparison, t0)
            if manifest.manifest_hash() != receipt.manifest_hash:
                return self.verdict(receipt, "fail", "manifest hash mismatch", 0.0, comparison, t0)
            graph = self.fetch_graph(manifest.graph_ref.sha256, manifest.graph_ref.uri)
            inputs = self.load_inputs(manifest.inputs, receipt.input_digests, comparison)
            outputs = evaluate(graph, inputs, manifest.params, bucket=self.bucket, device=self.config.device)
            replay_compute = time.time() - t0
            ok = True
            reasons: list[str] = []
            for ref in manifest.outputs:
                out_ok, out_cmp = self.compare_output(ref, outputs.get(ref.name), manifest)
                comparison["outputs"][ref.name] = out_cmp
                if not out_ok:
                    ok = False
                    reasons.append(f"{ref.name}: {out_cmp.get('reason')}")
            status = "pass" if ok else "fail"
            reason = "all outputs matched" if ok else "; ".join(reasons[:4])
            return self.verdict(receipt, status, reason, replay_compute, comparison, t0)
        except TimelockPending as e:
            comparison["crypto_pending"] = str(e)
            return self.verdict(receipt, "inconclusive", str(e), 0.0, comparison, t0)
        except ValueError as e:
            comparison["verification_error"] = str(e)
            return self.verdict(receipt, "fail", str(e), 0.0, comparison, t0)
        except Exception as e:
            comparison["error"] = repr(e)
            return self.verdict(receipt, "inconclusive", str(e), 0.0, comparison, t0)

    def find_manifest(self, receipt: JobReceiptV3) -> JobManifestV3:
        uri = self.bucket.uri_for_key(paths.job_manifest_key(self.config.netuid, receipt.run_id, receipt.job_id))
        return JobManifestV3.from_dict(self.bucket.get_json(uri))

    def verify_manifest_signature(self, manifest: JobManifestV3) -> bool:
        if not manifest.owner_signature:
            return False
        return verify_dict(manifest.unsigned_dict(), self.config.owner_secret, manifest.owner_signature)

    def verify_receipt_signature(self, receipt: JobReceiptV3) -> bool:
        if not receipt.miner_signature:
            return False
        return verify_dict(receipt.unsigned_dict(), self.config.miner_secret, receipt.miner_signature)

    def fetch_graph(self, sha: str, uri: str) -> Graph:
        cached = self.graph_cache.get(sha)
        if cached is not None:
            return cached
        graph = Graph.from_dict(json.loads(self.bucket.get(uri).decode("utf-8")))
        if graph.graph_id() != sha:
            raise ValueError("graph hash mismatch")
        self.graph_cache[sha] = graph
        return graph

    def load_inputs(
        self,
        refs: list[ArtifactRef],
        digests: list[ArtifactDigest],
        comparison: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        expected = {d.name: d for d in digests}
        inputs: dict[str, torch.Tensor] = {}
        for ref in refs:
            body = self.bucket.get(ref.uri)
            sha = hashlib.sha256(body).hexdigest()
            exp = expected.get(ref.name)
            comparison["inputs"][ref.name] = {
                "sha256": sha,
                "size_bytes": len(body),
                "matches_receipt": exp is None or exp.sha256 == sha or exp.envelope_sha256 == sha,
                "crypto_mode": ref.crypto.mode if ref.crypto else "none",
            }
            if exp is not None and exp.sha256 != sha and exp.envelope_sha256 != sha:
                raise ValueError(f"input changed: {ref.name}")
            body = decode_envelope(
                body,
                ref.crypto,
                verifier=HmacSigner(self.config.miner_secret),
                encryption_secret=self.config.encryption_secret,
                timelock_provider=self.config.timelock_provider,
            )
            inputs[ref.name] = tensor_io.decode_tensor(body).to(self.config.device)
        return inputs

    def compare_output(self, ref: ArtifactRef, expected_value: Any, manifest: JobManifestV3) -> tuple[bool, dict[str, Any]]:
        if expected_value is None:
            return False, {"status": "fail", "reason": "missing replay output"}
        if ref.uri.endswith(".json"):
            observed_body = decode_envelope(
                self.bucket.get(ref.uri),
                ref.crypto,
                verifier=HmacSigner(self.config.miner_secret),
                encryption_secret=self.config.encryption_secret,
                timelock_provider=self.config.timelock_provider,
            )
            observed_json = json.loads(observed_body.decode("utf-8"))
            observed = torch.as_tensor(observed_json.get("value"), device=self.config.device)
        else:
            observed_body = decode_envelope(
                self.bucket.get(ref.uri),
                ref.crypto,
                verifier=HmacSigner(self.config.miner_secret),
                encryption_secret=self.config.encryption_secret,
                timelock_provider=self.config.timelock_provider,
            )
            observed = tensor_io.decode_tensor(observed_body).to(self.config.device)
        if not isinstance(expected_value, torch.Tensor):
            return False, {"status": "fail", "reason": "non-tensor replay output"}
        expected = expected_value.detach().to(self.config.device)
        if list(observed.shape) != list(expected.shape):
            return False, {"status": "fail", "reason": "shape mismatch"}
        policy = manifest.verification_policy
        obs = observed.reshape(-1)
        exp = expected.reshape(-1)
        total = int(obs.numel())
        if 0 < policy.max_sample_elements < total:
            rng = random.Random(policy.sample_seed)
            idx = torch.as_tensor(rng.sample(range(total), policy.max_sample_elements), device=self.config.device)
            obs = obs.index_select(0, idx)
            exp = exp.index_select(0, idx)
        comparator = policy.comparator
        if comparator == "auto":
            comparator = "allclose" if expected.dtype.is_floating_point else "exact"
        if comparator == "exact":
            ok = bool(torch.equal(obs.cpu(), exp.cpu()))
            max_abs = 0.0 if ok else float((obs.to(torch.float32) - exp.to(torch.float32)).abs().max().item())
        else:
            ok = bool(torch.allclose(obs.to(torch.float32), exp.to(torch.float32), rtol=policy.rtol, atol=policy.atol))
            max_abs = 0.0 if obs.numel() == 0 else float((obs.to(torch.float32) - exp.to(torch.float32)).abs().max().item())
        return ok, {
            "status": "pass" if ok else "fail",
            "reason": "matched" if ok else "tensor mismatch",
            "comparator": comparator,
            "checked_elements": int(obs.numel()),
            "total_elements": total,
            "max_abs_error": max_abs,
            "crypto_mode": ref.crypto.mode if ref.crypto else "none",
        }

    def verdict(
        self,
        receipt: JobReceiptV3,
        status: str,
        reason: str,
        replay_compute_sec: float,
        comparison: dict[str, Any],
        t0: float,
    ) -> VerificationVerdictV3:
        estimated_cu = estimate_cu(receipt, replay_compute_sec)
        verdict = VerificationVerdictV3(
            verdict_id=f"{self.config.validator_hotkey}:{receipt.receipt_id}",
            receipt_id=receipt.receipt_id,
            manifest_hash=receipt.manifest_hash,
            job_id=receipt.job_id,
            run_id=receipt.run_id,
            miner_hotkey=receipt.worker.hotkey_ss58,
            validator_hotkey=self.config.validator_hotkey,
            status=status,
            reason=reason,
            estimated_cu=estimated_cu,
            replay_compute_sec=replay_compute_sec,
            checked_unix=time.time(),
            comparison=comparison,
        )
        return verdict.sign(self.config.validator_secret)


def estimate_cu(receipt: JobReceiptV3, replay_compute_sec: float = 0.0) -> float:
    compute = replay_compute_sec if replay_compute_sec > 0 else receipt.compute_sec
    return compute + (receipt.claimed_bytes_read + receipt.claimed_bytes_written) / 1_000_000_000.0


def summarize_scores(
    bucket: ObjectStore,
    *,
    netuid: int,
    run_id: str,
    window_id: str | None = None,
    validator_secret: str = "validator-dev-secret",
) -> dict[str, MinerScoreWindow]:
    window_id = window_id or f"run={run_id}"
    receipts: dict[str, JobReceiptV3] = {}
    for uri in bucket.list(bucket.uri_for_key(paths.receipts_prefix(netuid, run_id))):
        if uri.endswith(".json"):
            r = JobReceiptV3.from_dict(bucket.get_json(uri))
            receipts[r.receipt_id] = r
    verdicts: dict[str, VerificationVerdictV3] = {}
    for uri in bucket.list(bucket.uri_for_key(paths.verdicts_prefix(netuid, run_id))):
        if uri.endswith(".json"):
            v = VerificationVerdictV3.from_dict(bucket.get_json(uri))
            if not v.validator_signature or not verify_dict(v.unsigned_dict(), validator_secret, v.validator_signature):
                continue
            verdicts[v.receipt_id] = v
    windows: dict[str, MinerScoreWindow] = {}
    for receipt in receipts.values():
        hotkey = receipt.worker.hotkey_ss58
        w = windows.setdefault(hotkey, MinerScoreWindow(netuid=netuid, window_id=window_id, hotkey_ss58=hotkey))
        w.receipts += 1
        verdict = verdicts.get(receipt.receipt_id)
        cu = estimate_cu(receipt, verdict.replay_compute_sec if verdict else 0.0)
        if verdict is None:
            w.unsampled_cu += cu
        else:
            w.verdicts += 1
            if verdict.status == "pass":
                w.pass_cu += cu
            elif verdict.status == "fail":
                w.fail_cu += cu
            else:
                w.unsampled_cu += cu * 0.5
    for w in windows.values():
        checked = w.pass_cu + w.fail_cu
        if checked == 0:
            w.trust_multiplier = 0.25
        elif w.fail_cu > 0:
            w.trust_multiplier = max(0.0, (w.pass_cu - 2.0 * w.fail_cu) / checked)
        else:
            w.trust_multiplier = 1.0
        w.score = w.pass_cu + w.unsampled_cu * w.trust_multiplier
    bucket.put_json(
        bucket.uri_for_key(paths.scores_key(netuid, window_id)),
        {hotkey: window.to_dict() for hotkey, window in windows.items()},
    )
    return windows
