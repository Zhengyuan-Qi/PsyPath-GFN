"""Launch the nine selected external adaptations with recorded per-model settings."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--cache',type=Path,required=True)
p.add_argument('--contract',type=Path,required=True)
p.add_argument('--third-party',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
if a.output.exists() and any(a.output.iterdir()):
    p.error('Use an empty output directory.')
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'scripts')]))
for name in ['legacy','misa','cubemlp','self_mm','dmd','simsuf']:
    config=json.loads((ROOT/f'configs/external_{name}.json').read_text())['parameters']
    config.update(meter_root=str(a.cache.resolve()),contract_dir=str(a.contract.resolve()),
                  multibench_root=str(a.third_party.resolve()/'MultiBench'),
                  external_models_root=str(a.third_party.resolve()),
                  output=str(a.output.resolve()/name),tasks=['daic_depression','edaic_depression','edaic_ptsd'],
                  device='cpu',threads=8)
    cmd=[sys.executable,str(ROOT/'scripts/run_published_models_on_meter_cache.py')]
    for key,value in config.items():
        if value is None: continue
        cmd.append('--'+key.replace('_','-'))
        cmd.extend(map(str,value if isinstance(value,list) else [value]))
    print(subprocess.list2cmdline(cmd),flush=True)
    if not a.dry_run: subprocess.run(cmd,cwd=ROOT,env=env,check=True)
