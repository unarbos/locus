# Mining

Miners run one or more worker processes under a single hotkey. In v3, the
simple model is one worker per GPU:

```text
hotkey H
  worker H-gpu0 -> cuda:0
  worker H-gpu1 -> cuda:1
```

The orchestrator assigns jobs to a hotkey and, usually, a specific worker. A
worker polls the bucket, executes assigned manifests, writes outputs, and emits
a signed receipt.

## Environment

Use Doppler or your shell environment to provide bucket access:

```bash
export S3_BUCKET=...
export S3_REGION=us-east-1
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

For dev signatures, all parties must agree on the miner secret:

```bash
export LOCUS_MINER_SECRET=miner-dev-secret
```

In production subnet mode, this should be replaced or backed by wallet-bound
signing.

## Run A Single-GPU Miner

```bash
locus-v3 miner \
  --run-id RUN_ID \
  --hotkey MINER_HOTKEY \
  --devices cuda \
  --poll-interval 0.2
```

You can also pass bucket flags explicitly:

```bash
locus-v3 miner \
  --run-id RUN_ID \
  --hotkey MINER_HOTKEY \
  --devices cuda \
  --s3-bucket "$S3_BUCKET" \
  --s3-region "$S3_REGION"
```

## Run Multiple GPUs

Use a comma-separated device list:

```bash
locus-v3 miner \
  --run-id RUN_ID \
  --hotkey MINER_HOTKEY \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

The miner supervisor creates one worker per listed device. Scores aggregate
under the same hotkey.

## Heartbeats

Each worker writes a heartbeat under:

```text
v3/netuid=<N>/miners/<hotkey>/workers/<worker_id>/heartbeat.json
```

Heartbeats include `run_id`, so an orchestrator only schedules workers for the
current run and ignores stale workers from older runs.

## Receipts

After a job succeeds, the worker writes a signed receipt under:

```text
v3/netuid=<N>/receipts/<run_id>/hotkey=<H>/<job_id>/attempt=<A>.json
```

Receipts include:

- manifest hash
- input/output digests
- worker identity
- timing and byte telemetry
- miner signature

The validator replays receipts before giving work credit.

## Adversarial Test Modes

For testing only:

```bash
locus-v3 miner \
  --run-id RUN_ID \
  --hotkey BAD_MINER \
  --devices cuda \
  --fault-mode partial_corrupt \
  --fault-rate 1.0
```

Supported modes currently include:

- `partial_corrupt`
- `wrong_output`
- `skip_compute`

These modes should not be enabled in production.
