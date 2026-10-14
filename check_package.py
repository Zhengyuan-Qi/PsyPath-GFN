"""Data-free integrity and synthetic tests; does not reproduce clinical results."""
import compileall
import os
from pathlib import Path
import subprocess
import sys
import json
import time

root=Path(__file__).resolve().parent
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root),str(root/'scripts')]),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
assert compileall.compile_dir(str(root/'scripts'),quiet=1)
names=['test_meter_gfn','test_meter_mh_online_operator','test_alternating_psychometric_gfn',
       'test_humanomniv2_pooling_mtl','test_psypath_flat_gfn_control','test_psypath_alternating_structure_formal']
results=[]
for name in names:
    started=time.time()
    p=subprocess.run([sys.executable,str(root/'scripts'/f'{name}.py')],cwd=root,env=env,capture_output=True,text=True)
    results.append(dict(test=name,exit_code=p.returncode,seconds=time.time()-started,stdout=p.stdout,stderr=p.stderr))
    print(name,p.returncode,flush=True)
p=subprocess.run([sys.executable,str(root/'scripts/reinforce_corrected.py'),'--unit-test'],cwd=root,env=env,capture_output=True,text=True)
results.append(dict(test='corrected_entropy_gradient',exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
(root/'verification').mkdir(exist_ok=True)
(root/'verification/synthetic_tests.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
raise SystemExit(int(any(r['exit_code'] for r in results)))
