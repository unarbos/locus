"""Artifact crypto envelopes for Locus v3."""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass

from locus_core.protocol import ArtifactCryptoPolicy, ArtifactEnvelope, CryptoMode
from locus_core.signatures import HmacSigner, Signer, Verifier, canonical_json, sha256_hex


class TimelockPending(Exception):
    """Raised when a timelocked artifact cannot be decrypted yet."""


class DrandTimelockProvider:
    def encrypt(self, plaintext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        raise NotImplementedError

    def decrypt(self, ciphertext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        raise NotImplementedError


@dataclass
class MockDrandTimelockProvider(DrandTimelockProvider):
    revealed_round: int = 0
    secret: str = "mock-drand"

    def encrypt(self, plaintext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        return xor_crypt(plaintext, key=f"{self.secret}:{round_number}")

    def decrypt(self, ciphertext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        if self.revealed_round < round_number:
            raise TimelockPending(f"drand round {round_number} is not revealed")
        return xor_crypt(ciphertext, key=f"{self.secret}:{round_number}")


class BittensorTimelockProvider(DrandTimelockProvider):
    """Adapter seam for bittensor.extras.timelock.

    The exact helper API can vary by Bittensor version; fail explicitly until
    the deployed subnet pins the version and maps policy fields to the SDK.
    """

    def __init__(self) -> None:
        try:
            import bittensor.extras.timelock as timelock  # noqa: F401
        except Exception as e:
            raise RuntimeError("bittensor timelock helpers are unavailable") from e

    def encrypt(self, plaintext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        raise NotImplementedError("wire Bittensor timelock encrypt for the pinned SDK version")

    def decrypt(self, ciphertext: bytes, *, round_number: int, policy: ArtifactCryptoPolicy) -> bytes:
        raise NotImplementedError("wire Bittensor timelock decrypt for the pinned SDK version")


def default_policy(policy: ArtifactCryptoPolicy | None) -> ArtifactCryptoPolicy:
    return policy or ArtifactCryptoPolicy()


def encode_envelope(
    plaintext: bytes,
    policy: ArtifactCryptoPolicy | None,
    *,
    signer: Signer | None = None,
    encryption_secret: str = "locus-dev-encryption",
    timelock_provider: DrandTimelockProvider | None = None,
) -> bytes:
    policy = default_policy(policy)
    mode = str(policy.mode)
    if mode == CryptoMode.NONE.value:
        return plaintext

    ciphertext = plaintext
    if mode == CryptoMode.ENCRYPTED.value:
        ciphertext = xor_crypt(plaintext, key=policy.key_id or encryption_secret)
    elif mode == CryptoMode.DRAND_TIMELOCK.value:
        if policy.drand_round is None:
            raise ValueError("drand_timelock policy requires drand_round")
        provider = timelock_provider or MockDrandTimelockProvider(revealed_round=0)
        ciphertext = provider.encrypt(plaintext, round_number=int(policy.drand_round), policy=policy)
    elif mode != CryptoMode.SIGNED.value:
        raise ValueError(f"unknown crypto mode: {mode}")

    envelope = ArtifactEnvelope(
        crypto_mode=mode,
        payload_b64=base64.b64encode(ciphertext).decode("ascii"),
        plaintext_sha256=sha256_hex(plaintext),
        ciphertext_sha256=sha256_hex(ciphertext),
        signer=getattr(signer, "identity", None),
        cipher_suite=policy.cipher_suite,
        key_id=policy.key_id,
        drand_round=policy.drand_round,
        drand_chain_hash=policy.drand_chain_hash,
        drand_public_key=policy.drand_public_key,
    )
    if signer is not None:
        envelope.signature = signer.sign(canonical_json(envelope.signed_payload_dict()))
    return json.dumps(envelope.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_envelope(
    blob: bytes,
    policy: ArtifactCryptoPolicy | None,
    *,
    verifier: Verifier | None = None,
    encryption_secret: str = "locus-dev-encryption",
    timelock_provider: DrandTimelockProvider | None = None,
) -> bytes:
    policy = default_policy(policy)
    mode = str(policy.mode)
    if mode == CryptoMode.NONE.value:
        return blob

    envelope = ArtifactEnvelope.from_dict(json.loads(blob.decode("utf-8")))
    verify_artifact_envelope(envelope, policy, verifier=verifier)
    ciphertext = base64.b64decode(envelope.payload_b64.encode("ascii"))
    if sha256_hex(ciphertext) != envelope.ciphertext_sha256:
        raise ValueError("artifact ciphertext digest mismatch")

    if mode == CryptoMode.SIGNED.value:
        plaintext = ciphertext
    elif mode == CryptoMode.ENCRYPTED.value:
        plaintext = xor_crypt(ciphertext, key=policy.key_id or encryption_secret)
    elif mode == CryptoMode.DRAND_TIMELOCK.value:
        if policy.drand_round is None:
            raise ValueError("drand_timelock policy requires drand_round")
        provider = timelock_provider or MockDrandTimelockProvider(revealed_round=0)
        plaintext = provider.decrypt(ciphertext, round_number=int(policy.drand_round), policy=policy)
    else:
        raise ValueError(f"unknown crypto mode: {mode}")

    if sha256_hex(plaintext) != envelope.plaintext_sha256:
        raise ValueError("artifact plaintext digest mismatch")
    return plaintext


def verify_artifact_envelope(
    envelope: ArtifactEnvelope,
    expected_policy: ArtifactCryptoPolicy,
    *,
    verifier: Verifier | None = None,
) -> None:
    if envelope.crypto_mode != str(expected_policy.mode):
        raise ValueError(f"artifact crypto mode mismatch: {envelope.crypto_mode} != {expected_policy.mode}")
    if expected_policy.required_signer and envelope.signer != expected_policy.required_signer:
        raise ValueError("artifact signer mismatch")
    if expected_policy.mode in (CryptoMode.SIGNED.value, CryptoMode.ENCRYPTED.value, CryptoMode.DRAND_TIMELOCK.value):
        if not envelope.signature:
            raise ValueError("artifact signature missing")
        if verifier is not None and not verifier.verify(
            canonical_json(envelope.signed_payload_dict()),
            envelope.signature,
            envelope.signer,
        ):
            raise ValueError("artifact signature verification failed")


def artifact_digest_from_blob(name: str, uri: str, blob: bytes, policy: ArtifactCryptoPolicy | None) -> dict:
    policy = default_policy(policy)
    mode = str(policy.mode)
    if mode == CryptoMode.NONE.value:
        return {
            "sha256": sha256_hex(blob),
            "size_bytes": len(blob),
            "crypto_mode": mode,
        }
    envelope = ArtifactEnvelope.from_dict(json.loads(blob.decode("utf-8")))
    return {
        "sha256": sha256_hex(blob),
        "size_bytes": len(blob),
        "plaintext_sha256": envelope.plaintext_sha256,
        "ciphertext_sha256": envelope.ciphertext_sha256,
        "envelope_sha256": sha256_hex(blob),
        "signature": envelope.signature,
        "crypto_mode": envelope.crypto_mode,
    }


def xor_crypt(data: bytes, *, key: str) -> bytes:
    stream = hashlib.sha256(key.encode("utf-8")).digest()
    out = bytearray()
    for i, b in enumerate(data):
        if i and i % len(stream) == 0:
            stream = hashlib.sha256(stream).digest()
        out.append(b ^ stream[i % len(stream)])
    return bytes(out)
