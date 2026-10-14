"""Print main-model per-seed means and sample SD without exporting individuals."""
import argparse
import json
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('results',nargs='+',type=Path)
a=p.parse_args()
for path in a.results:
    d=json.loads(path.read_text(encoding='utf-8'))
    rows=[r['methods']['gflownet']['test'] for r in d['runs']]
    out={'task':d['task'],'seeds':len(rows),'sd_definition':'sample, ddof=1','metrics':{}}
    for section,keys in [('posterior',['weighted_f1','auroc','brier']),('severity',['ccc','mae'])]:
        for key in keys:
            values=[(r[section]['posterior'] if section=='severity' else r[section])[key] for r in rows]
            out['metrics'][key]={'mean':statistics.mean(values),'sd':statistics.stdev(values) if len(values)>1 else None}
    print(json.dumps(out,indent=2))
