"""Portable entry point for the paper's frozen-representation experiments."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TASKS = ['daic_depression', 'edaic_depression', 'edaic_ptsd']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment', choices=['main','no-pim','score-permutation','reinforce','fixed-views','pool-cls','pool-mtl'])
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--formal-root', type=Path, help='Main-model root containing task development_checkpoints; required for pooling')
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    p.add_argument('--dry-run', action='store_true')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use a new, empty output directory; existing runs are not overwritten.')
    env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'scripts')]))
    commands=[]
    if a.experiment.startswith('pool-'):
        if not a.formal_root:
            p.error('--formal-root is required for pooling')
        kind='cls' if a.experiment=='pool-cls' else 'mtl'
        commands.append([sys.executable,str(ROOT/f'scripts/run_humanomniv2_pooling_{kind}.py'),
            '--protocol',str(ROOT/f'configs/pooling_{kind}.json'),'--cache',str(a.cache.resolve()),
            '--contract',str(a.contract.resolve()),'--formal-root',str(a.formal_root.resolve()),'--output',str(a.output.resolve()),
            '--tasks',*a.tasks])
    else:
        script={'reinforce':'reinforce_corrected','fixed-views':'fixed_views'}.get(a.experiment,'validate_alternating_psychometric_gfn_cached')
        variant={'no-pim':'no_pim_no_ensr','score-permutation':'score_permutation_no_ensr'}.get(a.experiment,'no_ensr')
        for task in a.tasks:
            commands.append([sys.executable,str(ROOT/f'scripts/{script}.py'),
                '--meter-root',str(a.cache.resolve()),'--contract-dir',str(a.contract.resolve()),
                '--output',str(a.output.resolve()/task),'--task',task,'--split-policy','official',
                '--variant',variant,'--selector-method','reinforce' if a.experiment=='reinforce' else 'gflownet',
                '--policy','legacy','--seeds','7','17','27','41','53','--device','cpu','--threads','6',
                '--cycles','3','--epochs-per-cycle','6','--hidden-dim','64','--phenotype-count','4',
                '--minimum-events','2','--maximum-events','16','--maximum-fraction','0.3',
                '--operator-trajectories','4','--selector-trajectories','4','--development-samples','16',
                '--test-samples','48','--measurement-components','48','--learning-rate','0.0003',
                '--weight-decay','0.0001','--gradient-accumulation','4','--pim-reward-weight','0.3',
                '--compactness-weight','0.05','--reward-temperature','0.5'])
    for cmd in commands:
        print(subprocess.list2cmdline(cmd),flush=True)
        if not a.dry_run:
            subprocess.run(cmd,cwd=ROOT,env=env,check=True)

if __name__=='__main__':
    main()
