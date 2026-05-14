"""Bittensor adapter for Locus v3 validators.

The adapter is deliberately isolated from the runtime so no-chain fleet tests
can use the same validator with `dry_run=True`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeightUpdate:
    uids: list[int]
    weights: list[float]


class BittensorAdapter:
    def __init__(
        self,
        *,
        netuid: int,
        wallet_name: str | None = None,
        hotkey_name: str | None = None,
        network: str | None = None,
        dry_run: bool = True,
    ) -> None:
        self.netuid = int(netuid)
        self.wallet_name = wallet_name
        self.hotkey_name = hotkey_name
        self.network = network
        self.dry_run = dry_run
        self._bt = None
        self._wallet = None
        self._subtensor = None
        if not dry_run:
            import bittensor as bt
            self._bt = bt
            self._wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
            self._subtensor = bt.Subtensor(network=network) if network else bt.Subtensor()

    def hotkey_to_uid(self) -> dict[str, int]:
        if self.dry_run:
            return {}
        metagraph = self._subtensor.metagraph(self.netuid)
        return {hotkey: int(uid) for uid, hotkey in enumerate(metagraph.hotkeys)}

    def publish_weights(self, scores: dict[str, float]) -> WeightUpdate:
        hotkey_to_uid = self.hotkey_to_uid()
        normalized = self.normalize_scores(scores)
        if self.dry_run:
            ordered = sorted(normalized.items())
            update = WeightUpdate(uids=list(range(len(ordered))), weights=[v for _h, v in ordered])
            print({"dry_run_set_weights": update.__dict__, "hotkeys": [h for h, _v in ordered]})
            return update
        missing = sorted(set(normalized) - set(hotkey_to_uid))
        if missing:
            print({"dropped_unknown_hotkeys": missing})
        pairs = [(hotkey_to_uid[h], score) for h, score in normalized.items() if h in hotkey_to_uid]
        pairs.sort()
        uids = [uid for uid, _score in pairs]
        weights = [float(score) for _uid, score in pairs]
        if not uids:
            raise RuntimeError("no score hotkeys map to current metagraph UIDs")
        result = self._subtensor.set_weights(
            wallet=self._wallet,
            netuid=self.netuid,
            uids=uids,
            weights=weights,
        )
        print({"set_weights": result, "uids": uids, "weights": weights})
        return WeightUpdate(uids=uids, weights=weights)

    @staticmethod
    def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        clean = {k: max(0.0, float(v)) for k, v in scores.items()}
        if not clean:
            return {}
        total = sum(clean.values())
        if total <= 0.0:
            equal = 1.0 / len(clean)
            return {k: equal for k in sorted(clean)}
        return {k: v / total for k, v in clean.items()}
