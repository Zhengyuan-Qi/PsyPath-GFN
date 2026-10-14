"""Locked checkpoint inference and CPU timing; no fitting or model selection."""
import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get('PSYPATH_DATA_ROOT', str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import validate_alternating_psychometric_gfn_cached as formal
from scripts import audit_psypath_reward_flow_faithfulness as audit
from scripts.alternating_psychometric_gfn import evidence_features
from scripts.meter_gfn import random_matched_mask

TASKS = ('daic_depression', 'edaic_depression', 'edaic_ptsd')
SEEDS = (7, 17, 27, 41, 53)
KS = (1, 2, 4, 8, 16, 48)


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def run(task, seed, smoke, out, cache=None, contract=None, checkpoint=None, skip_timing=False):
    formal.configure_runtime(1)
    root = cache or ROOT/'outputs/meter_mh_official_e2e_pipeline_20260813/official_cache'
    labels = contract or ROOT/'data/clinical_label_contract_v1'
    formal.validate_contract(labels)
    split = 'dev' if smoke else 'test'
    raw = formal.load_records(root, labels, task, allowed_splits=(split,), split_policy='official')[split]
    formal.attach_temporal({split: raw}, root, formal.TASKS[task]['dataset'])
    expected = (35 if task == TASKS[0] else 56) if smoke else (47 if task == TASKS[0] else 56)
    assert len(raw) == expected
    if smoke:
        raw = raw[:2]
    cp = checkpoint or ROOT/f'outputs/psychometric_gfn_formal_suite_v1_20260818/pim_gflownet/{task}/checkpoints/{task}/gflownet/seed_{seed}.pt'
    ck = formal.load_trusted_checkpoint(cp, torch.device('cpu'))
    assert ck['task'] == task and ck['seed'] == seed and ck['variant'] == 'no_ensr'
    config = audit.checkpoint_args(ck)
    config.exploration_epsilon = 0.
    records = formal.prepare_records(raw, ck['event_projections'], torch.device('cpu'))
    model = formal.new_model(records[0], config, torch.device('cpu'), task_name=task)
    model.load_state_dict(ck['gflownet_state'], strict=True)
    model.eval()
    op = ck['psychometric_operator']
    # Preserve the published test draw identities exactly at K=48.
    posterior_seed = formal.stable_seed(formal.stable_seed(task, seed, 'test'), 'posterior')
    rows = []
    with torch.inference_mode():
        for record in records:
            enc = model.encode(record['events'])
            learned, random = [], []
            for j in range(48):
                t = formal.sample_trajectory(model, record, config,
                    formal.stable_seed(posterior_seed, record['pid'], j), gradient=False, encoding=enc)
                r = random_matched_mask(t.mask, formal.stable_seed(posterior_seed, record['pid'], j, 'random'))
                assert torch.equal(t.mask.sum(0), r.sum(0))
                assert t.mask.any(1).sum() == r.any(1).sum()
                learned.append(t.mask.numpy())
                random.append(r.numpy())
            pred = op.predict(np.stack([evidence_features(record['projected_np'], m) for m in learned+random]))
            assert all(np.isfinite(pred[k]).all() for k in ('probability','severity'))
            rows.append(dict(pid=str(record['pid']), label=record['label'], score=record['score'],
                n_events=len(record['events']),
                learned_probability=pred['probability'][:48].tolist(), random_probability=pred['probability'][48:].tolist(),
                learned_severity=pred['severity'][:48].tolist(), random_severity=pred['severity'][48:].tolist()))
    save(out/f'{task}_seed_{seed}.json', dict(task=task, seed=seed, split=split,
         label_profile=formal.TASKS[task]['binary_profile'], checkpoint=str(cp),
         checkpoint_sha256=formal.sha256_file(cp), threshold=ck['threshold'], rows=rows,
         policy_parameters=sum(p.numel() for p in model.parameters()),
         policy_parameter_shapes={n:list(p.shape) for n,p in model.named_parameters()},
         pim_pca_shape=list(op.projection.reducer.components_.shape),
         pim_regression_coefficients=int(op.severity_model.coef_.size+np.asarray(op.severity_model.intercept_).size),
         pim_calibration_coefficients=int(op.calibrator.coef_.size+op.calibrator.intercept_.size)))
    if seed == 7 and not skip_timing:
        # One shared CPU thread; preloaded frozen representations. No GPU timing claims.
        timings = []
        with torch.inference_mode():
            for record in records:
                model.encode(record['events'])
                # Warm the complete path and measurement operations for this participant.
                enc = model.encode(record['events'])
                t = formal.sample_trajectory(model, record, config, 11, gradient=False, encoding=enc)
                op.predict(evidence_features(record['projected_np'], t.mask.numpy()))
                for k in KS:
                    for rep in range(3):
                        a = time.perf_counter()
                        enc = model.encode(record['events'])
                        b = time.perf_counter()
                        masks = [formal.sample_trajectory(model, record, config,
                            formal.stable_seed('timing-v47',task,record['pid'],rep,j),
                            gradient=False, encoding=enc).mask.numpy() for j in range(k)]
                        c = time.perf_counter()
                        f = np.stack([evidence_features(record['projected_np'],m) for m in masks])
                        pred = op.predict(f)
                        float(pred['probability'].mean()); float(pred['severity'].mean())
                        end = time.perf_counter()
                        timings.append(dict(pid=str(record['pid']), k=k, repeat=rep,
                            encoding_ms=1000*(b-a), sampling_ms=1000*(c-b),
                            measurement_ms=1000*(end-c), total_ms=1000*(end-a)))
        save(out/f'{task}_timing.json',dict(task=task, split=split, n=len(records),
            repeats=3, threads=1, scope='Preloaded frozen representations; shared event encoder + GFN + PIM. Excludes HumanOmniV2, I/O and preprocessing.',
            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            torch_version=torch.__version__, python=platform.python_version(),
            cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), platform.processor()),
            gpu_used=False, timings=timings))
    print(json.dumps(dict(task=task,seed=seed,status='complete',n=len(rows))),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=TASKS,required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--contract',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--skip-timing',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/f'{args.task}_seed_{args.seed}.json').exists()
    run(args.task,args.seed,args.smoke,args.out,args.cache,args.contract,args.checkpoint,args.skip_timing)
