"""Validator neuron facade for Locus v3."""
from __future__ import annotations

from dataclasses import dataclass

from locus_runtime.storage import ObjectStore
from .subnet import BittensorAdapter, WeightUpdate
from .verifier import ReplayVerifier, ValidatorConfig, summarize_scores


@dataclass
class ValidatorNeuronConfig:
    netuid: int
    run_id: str
    validator_hotkey: str
    validator_secret: str = "validator-dev-secret"
    owner_secret: str = "owner-dev-secret"
    miner_secret: str = "miner-dev-secret"
    device: str = "cpu"
    sample_rate: float = 1.0
    encryption_secret: str = "locus-dev-encryption"
    timelock_provider: object | None = None
    dry_run_weights: bool = True
    wallet_name: str | None = None
    hotkey_name: str | None = None
    network: str | None = None


class ValidatorNeuron:
    def __init__(self, *, bucket: ObjectStore, config: ValidatorNeuronConfig) -> None:
        self.bucket = bucket
        self.config = config
        self.verifier = ReplayVerifier(
            bucket=bucket,
            config=ValidatorConfig(
                netuid=config.netuid,
                run_id=config.run_id,
                validator_hotkey=config.validator_hotkey,
                validator_secret=config.validator_secret,
                owner_secret=config.owner_secret,
                miner_secret=config.miner_secret,
                device=config.device,
                sample_rate=config.sample_rate,
                encryption_secret=config.encryption_secret,
                timelock_provider=config.timelock_provider,
            ),
        )
        self.subnet = BittensorAdapter(
            netuid=config.netuid,
            wallet_name=config.wallet_name,
            hotkey_name=config.hotkey_name,
            network=config.network,
            dry_run=config.dry_run_weights,
        )

    def run_once(self, *, max_receipts: int | None = None, publish_weights: bool = False) -> dict:
        checked = self.verifier.run_once(max_receipts=max_receipts)
        windows = summarize_scores(
            self.bucket,
            netuid=self.config.netuid,
            run_id=self.config.run_id,
            window_id=f"run={self.config.run_id}",
            validator_secret=self.config.validator_secret,
        )
        scores = {hotkey: window.score for hotkey, window in windows.items()}
        update: WeightUpdate | None = None
        if publish_weights:
            update = self.subnet.publish_weights(scores)
        return {
            "checked": checked,
            "scores": {k: v.to_dict() for k, v in windows.items()},
            "weight_update": update.__dict__ if update is not None else None,
        }
