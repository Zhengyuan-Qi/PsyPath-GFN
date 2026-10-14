"""Recalculate F1/AUROC from private saved predictions; print aggregates only."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('results',nargs='+',type=Path)
a=p.parse_args()
for path in a.results:
    d=json.loads(path.read_text())
    verified=[]
    for run in d['runs']:
        test=run['methods']['gflownet']['test']
        y=np.asarray(test['labels']);prob=np.asarray(test['probability']['posterior'])
        values={'weighted_f1':f1_score(y,prob>=test['threshold'],average='weighted'),
                'auroc':roc_auc_score(y,prob)}
        for metric,value in values.items():
            if not np.isclose(value,test['posterior'][metric],rtol=0,atol=1e-12):
                raise RuntimeError(f"Mismatch: {d['task']} seed {run['seed']} {metric}")
        verified.append({'seed':run['seed'],**values})
    print(json.dumps({'task':d['task'],'verified':verified},indent=2))
