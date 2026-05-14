"""Small deterministic signing helpers.

For no-chain tests we use shared-secret HMAC signatures. In subnet mode these
records can additionally be signed by Bittensor wallets; the payload hashing
here remains the canonical bytes-to-sign surface.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_dict(value: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(value))


def sign_dict(value: dict[str, Any], secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), canonical_json(value), hashlib.sha256).hexdigest()


def verify_dict(value: dict[str, Any], secret: str, signature: str) -> bool:
    return hmac.compare_digest(sign_dict(value, secret), signature)
