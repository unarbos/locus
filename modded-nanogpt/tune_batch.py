import os
import re
import json
import sys
import subprocess
import time
import optuna
from heavyball.helpers import HEBOSampler

VAL_RE = re.compile(r'val_loss\s+(\d+\.\d+)')


def parse_val(text):
    m = VAL_RE.findall(text)
    return float(m[-1]) if m else float('inf')


def run(lr, loco_lr, k, batch, n_steps, staged=None, seed=42):
    if staged is None:
        staged = 1 if k > 0 else 0
    env = os.environ.copy()
    env.update({
        'LR': f'{lr:.6e}',
        'LOCO_LR': f'{loco_lr:.6e}',
        'LOCO_STEPS': str(k),
        'STAGED': str(staged),
        'SIGN': '1',
        'BATCH_SEQ': str(batch),
        'N_STEPS': str(n_steps),
        'SEED': str(seed),
        'LOG_EVERY': '50',
    })
    t0 = time.time()
    r = subprocess.run(
        ['torchrun', '--standalone', '--nproc_per_node=8', 'train_t3.py'],
        env=env, capture_output=True, text=True, timeout=14400,
    )
    val = parse_val(r.stdout + r.stderr)
    dt = time.time() - t0
    return val, dt


def tune(batch, k, n_trials, n_steps, log_path):
    has_loco = k > 0
    space = {'lr': optuna.distributions.FloatDistribution(1e-6, 1, log=True)}
    if has_loco:
        space['loco_lr'] = optuna.distributions.FloatDistribution(1e-6, 1, log=True)

    sampler = HEBOSampler(search_space=space, seed=0)
    study = optuna.create_study(sampler=sampler, direction='minimize')

    def obj(trial):
        lr = trial.suggest_float('lr', 1e-6, 1, log=True)
        loco_lr = trial.suggest_float('loco_lr', 1e-6, 1, log=True) if has_loco else 0.0
        v, dt = run(lr, loco_lr, k, batch, n_steps)
        with open(log_path, 'a') as f:
            f.write(f"trial b={batch} k={k} lr={lr:.4e} loco_lr={loco_lr:.4e} val={v:.4f} t={dt:.1f}s\n")
        return v

    study.optimize(obj, n_trials=n_trials, gc_after_trial=True)
    return study.best_params, study.best_value


def main():
    BATCHES = [int(x) for x in os.environ.get('BATCHES', '8,64,256,1024').split(',')]
    K = int(os.environ.get('K', '4'))
    N_TUNE = int(os.environ.get('N_TUNE', '50'))
    N_TRIALS = int(os.environ.get('N_TRIALS', '6'))
    OUT = os.environ.get('OUT', 'tune_results.json')
    LOG = os.environ.get('LOG', 'tune.log')

    results = {}
    for batch in BATCHES:
        for name, k in [('baseline', 0), (f'locoK{K}', K)]:
            print(f"\n=== TUNE batch={batch} {name} ===", flush=True)
            with open(LOG, 'a') as f:
                f.write(f"\n=== TUNE batch={batch} {name} ===\n")
            params, val = tune(batch, k, N_TRIALS, N_TUNE, LOG)
            results[f"b{batch}_{name}"] = {'params': params, 'val': val, 'k': k, 'batch': batch}
            print(f"BEST batch={batch} {name}: {params} val={val:.4f}", flush=True)
            with open(OUT, 'w') as f:
                json.dump(results, f, indent=2)

    print("\n=== TUNING DONE ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
