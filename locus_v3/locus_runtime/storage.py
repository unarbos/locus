"""Object-store abstractions for Locus v3."""
from __future__ import annotations

from typing import Protocol

from locus_core.compat import ensure_v2_path

ensure_v2_path()

from locus.storage import LocalBucket, S3Bucket, join_uri, parse_uri  # noqa: F401


class ObjectStore(Protocol):
    bucket: str

    def uri_for_key(self, key: str, *, bucket: str | None = None) -> str: ...
    def put(self, uri: str, data: bytes) -> None: ...
    def get(self, uri: str) -> bytes: ...
    def exists(self, uri: str) -> bool: ...
    def delete(self, uri: str) -> None: ...
    def list(self, prefix_uri: str) -> list[str]: ...
    def get_json(self, uri: str) -> dict: ...
    def put_json(self, uri: str, value: dict | list) -> None: ...


def open_local_bucket(root: str, bucket: str) -> LocalBucket:
    return LocalBucket(root=root, bucket=bucket)
