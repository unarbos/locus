"""V3 wire protocol records.

These are the bucket and subnet-facing records. The tensor graphs themselves
remain content-addressed Locus IR graphs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .signatures import digest_dict, sign_dict


class CryptoMode(str, Enum):
    NONE = "none"
    SIGNED = "signed"
    ENCRYPTED = "encrypted"
    DRAND_TIMELOCK = "drand_timelock"


@dataclass
class ArtifactCryptoPolicy:
    mode: str = CryptoMode.NONE.value
    required_signer: str | None = None
    recipient: str | None = None
    cipher_suite: str = "xor-dev-v1"
    key_id: str | None = None
    drand_round: int | None = None
    drand_chain_hash: str | None = None
    drand_public_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mode": str(self.mode),
            "cipher_suite": self.cipher_suite,
        }
        for key in (
            "required_signer",
            "recipient",
            "key_id",
            "drand_round",
            "drand_chain_hash",
            "drand_public_key",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "ArtifactCryptoPolicy":
        d = dict(d or {})
        return ArtifactCryptoPolicy(
            mode=d.get("mode", CryptoMode.NONE.value),
            required_signer=d.get("required_signer"),
            recipient=d.get("recipient"),
            cipher_suite=d.get("cipher_suite", "xor-dev-v1"),
            key_id=d.get("key_id"),
            drand_round=d.get("drand_round"),
            drand_chain_hash=d.get("drand_chain_hash"),
            drand_public_key=d.get("drand_public_key"),
        )


@dataclass
class ArtifactEnvelope:
    crypto_mode: str
    payload_b64: str
    plaintext_sha256: str
    ciphertext_sha256: str
    signer: str | None = None
    signature: str | None = None
    cipher_suite: str | None = None
    key_id: str | None = None
    drand_round: int | None = None
    drand_chain_hash: str | None = None
    drand_public_key: str | None = None
    schema_version: int = 1

    def signed_payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "crypto_mode": self.crypto_mode,
            "payload_b64": self.payload_b64,
            "plaintext_sha256": self.plaintext_sha256,
            "ciphertext_sha256": self.ciphertext_sha256,
            "signer": self.signer,
            "cipher_suite": self.cipher_suite,
            "key_id": self.key_id,
            "drand_round": self.drand_round,
            "drand_chain_hash": self.drand_chain_hash,
            "drand_public_key": self.drand_public_key,
        }

    def to_dict(self) -> dict[str, Any]:
        out = self.signed_payload_dict()
        out["signature"] = self.signature
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ArtifactEnvelope":
        return ArtifactEnvelope(
            schema_version=int(d.get("schema_version", 1)),
            crypto_mode=d.get("crypto_mode", CryptoMode.NONE.value),
            payload_b64=d["payload_b64"],
            plaintext_sha256=d["plaintext_sha256"],
            ciphertext_sha256=d["ciphertext_sha256"],
            signer=d.get("signer"),
            signature=d.get("signature"),
            cipher_suite=d.get("cipher_suite"),
            key_id=d.get("key_id"),
            drand_round=d.get("drand_round"),
            drand_chain_hash=d.get("drand_chain_hash"),
            drand_public_key=d.get("drand_public_key"),
        )


@dataclass
class ArtifactRef:
    name: str
    uri: str
    sha256: str | None = None
    size_bytes: int | None = None
    crypto: ArtifactCryptoPolicy | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "uri": self.uri}
        if self.sha256 is not None:
            out["sha256"] = self.sha256
        if self.size_bytes is not None:
            out["size_bytes"] = int(self.size_bytes)
        if self.crypto is not None:
            out["crypto"] = self.crypto.to_dict()
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ArtifactRef":
        return ArtifactRef(
            name=d["name"],
            uri=d["uri"],
            sha256=d.get("sha256"),
            size_bytes=d.get("size_bytes"),
            crypto=ArtifactCryptoPolicy.from_dict(d.get("crypto")) if d.get("crypto") else None,
        )


@dataclass
class ArtifactDigest:
    name: str
    uri: str
    sha256: str
    size_bytes: int
    plaintext_sha256: str | None = None
    ciphertext_sha256: str | None = None
    envelope_sha256: str | None = None
    signature: str | None = None
    crypto_mode: str = CryptoMode.NONE.value

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
            "crypto_mode": self.crypto_mode,
        }
        for key in ("plaintext_sha256", "ciphertext_sha256", "envelope_sha256", "signature"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ArtifactDigest":
        return ArtifactDigest(
            name=d["name"],
            uri=d["uri"],
            sha256=d["sha256"],
            size_bytes=int(d["size_bytes"]),
            plaintext_sha256=d.get("plaintext_sha256"),
            ciphertext_sha256=d.get("ciphertext_sha256"),
            envelope_sha256=d.get("envelope_sha256"),
            signature=d.get("signature"),
            crypto_mode=d.get("crypto_mode", CryptoMode.NONE.value),
        )


@dataclass
class GraphRef:
    sha256: str
    uri: str

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "uri": self.uri}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GraphRef":
        return GraphRef(sha256=d["sha256"], uri=d["uri"])


@dataclass
class MinerIdentity:
    netuid: int
    hotkey_ss58: str
    uid: int | None = None
    endpoint: str | None = None
    commitment_hash: str | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "netuid": int(self.netuid),
            "hotkey_ss58": self.hotkey_ss58,
            "uid": self.uid,
            "endpoint": self.endpoint,
            "commitment_hash": self.commitment_hash,
            "capabilities": dict(self.capabilities),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "MinerIdentity":
        return MinerIdentity(
            netuid=int(d["netuid"]),
            hotkey_ss58=d["hotkey_ss58"],
            uid=d.get("uid"),
            endpoint=d.get("endpoint"),
            commitment_hash=d.get("commitment_hash"),
            capabilities=dict(d.get("capabilities") or {}),
        )


@dataclass
class WorkerIdentity:
    hotkey_ss58: str
    worker_id: str
    host_id: str
    gpu_index: int | None
    session_nonce: str
    software_hash: str = "dev"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hotkey_ss58": self.hotkey_ss58,
            "worker_id": self.worker_id,
            "host_id": self.host_id,
            "gpu_index": self.gpu_index,
            "session_nonce": self.session_nonce,
            "software_hash": self.software_hash,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "WorkerIdentity":
        return WorkerIdentity(
            hotkey_ss58=d["hotkey_ss58"],
            worker_id=d["worker_id"],
            host_id=d.get("host_id", "unknown"),
            gpu_index=d.get("gpu_index"),
            session_nonce=d.get("session_nonce", "unknown"),
            software_hash=d.get("software_hash", "dev"),
        )


@dataclass
class VerificationPolicy:
    method: str = "replay_ir_v1"
    comparator: str = "auto"
    rtol: float = 1e-3
    atol: float = 1e-4
    max_sample_elements: int = 4096
    sample_seed: int = 0
    critical: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "comparator": self.comparator,
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "max_sample_elements": int(self.max_sample_elements),
            "sample_seed": int(self.sample_seed),
            "critical": bool(self.critical),
        }

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "VerificationPolicy":
        d = dict(d or {})
        return VerificationPolicy(
            method=d.get("method", "replay_ir_v1"),
            comparator=d.get("comparator", "auto"),
            rtol=float(d.get("rtol", 1e-3)),
            atol=float(d.get("atol", 1e-4)),
            max_sample_elements=int(d.get("max_sample_elements", 4096)),
            sample_seed=int(d.get("sample_seed", 0)),
            critical=bool(d.get("critical", False)),
        )


@dataclass
class JobManifestV3:
    job_id: str
    run_id: str
    step_id: int
    kind: str
    graph_ref: GraphRef
    params: dict[str, Any]
    inputs: list[ArtifactRef]
    outputs: list[ArtifactRef]
    assigned_hotkey: str
    assigned_worker: str | None
    attempt: int
    deadline_unix: int
    created_unix: int
    verification_policy: VerificationPolicy = field(default_factory=VerificationPolicy)
    owner_signature: str | None = None
    schema_version: int = 3

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "job_id": self.job_id,
            "run_id": self.run_id,
            "step_id": int(self.step_id),
            "kind": self.kind,
            "graph_ref": self.graph_ref.to_dict(),
            "params": dict(self.params),
            "inputs": [r.to_dict() for r in self.inputs],
            "outputs": [r.to_dict() for r in self.outputs],
            "assigned_hotkey": self.assigned_hotkey,
            "assigned_worker": self.assigned_worker,
            "attempt": int(self.attempt),
            "deadline_unix": int(self.deadline_unix),
            "created_unix": int(self.created_unix),
            "verification_policy": self.verification_policy.to_dict(),
        }

    def manifest_hash(self) -> str:
        return digest_dict(self.unsigned_dict())

    def sign(self, secret: str) -> "JobManifestV3":
        self.owner_signature = sign_dict(self.unsigned_dict(), secret)
        return self

    def to_dict(self) -> dict[str, Any]:
        out = self.unsigned_dict()
        out["manifest_hash"] = self.manifest_hash()
        out["owner_signature"] = self.owner_signature
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "JobManifestV3":
        return JobManifestV3(
            schema_version=int(d.get("schema_version", 3)),
            job_id=d["job_id"],
            run_id=d["run_id"],
            step_id=int(d["step_id"]),
            kind=d["kind"],
            graph_ref=GraphRef.from_dict(d["graph_ref"]),
            params=dict(d.get("params") or {}),
            inputs=[ArtifactRef.from_dict(x) for x in d["inputs"]],
            outputs=[ArtifactRef.from_dict(x) for x in d["outputs"]],
            assigned_hotkey=d["assigned_hotkey"],
            assigned_worker=d.get("assigned_worker"),
            attempt=int(d.get("attempt", 0)),
            deadline_unix=int(d["deadline_unix"]),
            created_unix=int(d["created_unix"]),
            verification_policy=VerificationPolicy.from_dict(d.get("verification_policy")),
            owner_signature=d.get("owner_signature"),
        )


@dataclass
class JobReceiptV3:
    receipt_id: str
    manifest_hash: str
    job_id: str
    run_id: str
    step_id: int
    kind: str
    worker: WorkerIdentity
    input_digests: list[ArtifactDigest]
    output_digests: list[ArtifactDigest]
    started_unix: float
    finished_unix: float
    compute_sec: float
    claimed_bytes_read: int
    claimed_bytes_written: int
    miner_signature: str | None = None
    schema_version: int = 3

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "receipt_id": self.receipt_id,
            "manifest_hash": self.manifest_hash,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "step_id": int(self.step_id),
            "kind": self.kind,
            "worker": self.worker.to_dict(),
            "input_digests": [d.to_dict() for d in self.input_digests],
            "output_digests": [d.to_dict() for d in self.output_digests],
            "started_unix": float(self.started_unix),
            "finished_unix": float(self.finished_unix),
            "compute_sec": float(self.compute_sec),
            "claimed_bytes_read": int(self.claimed_bytes_read),
            "claimed_bytes_written": int(self.claimed_bytes_written),
        }

    def sign(self, secret: str) -> "JobReceiptV3":
        self.miner_signature = sign_dict(self.unsigned_dict(), secret)
        return self

    def to_dict(self) -> dict[str, Any]:
        out = self.unsigned_dict()
        out["miner_signature"] = self.miner_signature
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "JobReceiptV3":
        return JobReceiptV3(
            schema_version=int(d.get("schema_version", 3)),
            receipt_id=d["receipt_id"],
            manifest_hash=d["manifest_hash"],
            job_id=d["job_id"],
            run_id=d["run_id"],
            step_id=int(d["step_id"]),
            kind=d["kind"],
            worker=WorkerIdentity.from_dict(d["worker"]),
            input_digests=[ArtifactDigest.from_dict(x) for x in d.get("input_digests", [])],
            output_digests=[ArtifactDigest.from_dict(x) for x in d.get("output_digests", [])],
            started_unix=float(d.get("started_unix", 0.0)),
            finished_unix=float(d.get("finished_unix", 0.0)),
            compute_sec=float(d.get("compute_sec", 0.0)),
            claimed_bytes_read=int(d.get("claimed_bytes_read", 0)),
            claimed_bytes_written=int(d.get("claimed_bytes_written", 0)),
            miner_signature=d.get("miner_signature"),
        )


@dataclass
class VerificationVerdictV3:
    verdict_id: str
    receipt_id: str
    manifest_hash: str
    job_id: str
    run_id: str
    miner_hotkey: str
    validator_hotkey: str
    status: str
    reason: str
    estimated_cu: float
    replay_compute_sec: float
    checked_unix: float
    comparison: dict[str, Any] = field(default_factory=dict)
    validator_signature: str | None = None
    schema_version: int = 3

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "verdict_id": self.verdict_id,
            "receipt_id": self.receipt_id,
            "manifest_hash": self.manifest_hash,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "miner_hotkey": self.miner_hotkey,
            "validator_hotkey": self.validator_hotkey,
            "status": self.status,
            "reason": self.reason,
            "estimated_cu": float(self.estimated_cu),
            "replay_compute_sec": float(self.replay_compute_sec),
            "checked_unix": float(self.checked_unix),
            "comparison": dict(self.comparison),
        }

    def sign(self, secret: str) -> "VerificationVerdictV3":
        self.validator_signature = sign_dict(self.unsigned_dict(), secret)
        return self

    def to_dict(self) -> dict[str, Any]:
        out = self.unsigned_dict()
        out["validator_signature"] = self.validator_signature
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "VerificationVerdictV3":
        return VerificationVerdictV3(
            schema_version=int(d.get("schema_version", 3)),
            verdict_id=d["verdict_id"],
            receipt_id=d["receipt_id"],
            manifest_hash=d["manifest_hash"],
            job_id=d["job_id"],
            run_id=d["run_id"],
            miner_hotkey=d["miner_hotkey"],
            validator_hotkey=d["validator_hotkey"],
            status=d["status"],
            reason=d.get("reason", ""),
            estimated_cu=float(d.get("estimated_cu", 0.0)),
            replay_compute_sec=float(d.get("replay_compute_sec", 0.0)),
            checked_unix=float(d.get("checked_unix", 0.0)),
            comparison=dict(d.get("comparison") or {}),
            validator_signature=d.get("validator_signature"),
        )


@dataclass
class MinerScoreWindow:
    netuid: int
    window_id: str
    hotkey_ss58: str
    receipts: int = 0
    verdicts: int = 0
    pass_cu: float = 0.0
    fail_cu: float = 0.0
    unsampled_cu: float = 0.0
    trust_multiplier: float = 1.0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "netuid": int(self.netuid),
            "window_id": self.window_id,
            "hotkey_ss58": self.hotkey_ss58,
            "receipts": int(self.receipts),
            "verdicts": int(self.verdicts),
            "pass_cu": float(self.pass_cu),
            "fail_cu": float(self.fail_cu),
            "unsampled_cu": float(self.unsampled_cu),
            "trust_multiplier": float(self.trust_multiplier),
            "score": float(self.score),
        }
