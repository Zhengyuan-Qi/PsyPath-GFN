"""Single online entry point: original interview -> individual screening output."""
import argparse,json,os
from pathlib import Path
import torch
from online_pipeline import read_input,run
from scripts.validate_alternating_psychometric_gfn_cached import configure_runtime

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--backbone',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cuda'],default='cuda')
    p.add_argument('--threads',type=int,default=6)
    a=p.parse_args()
    if a.output.exists(): p.error('Output already exists; choose a new file')
    configure_runtime(min(a.threads,32))
    if a.device=='cuda':
        if not torch.cuda.is_available(): p.error('CUDA unavailable')
        free,total=torch.cuda.mem_get_info()
        if free<30*1024**3: p.error('Less than 30 GiB free on selected GPU; choose another GPU')
        os.environ.setdefault('EPCM_CUDA_MEMORY_FRACTION',str(min(.60,(free-6*1024**3)/total)))
    result=run(read_input(a.input),a.model_dir,a.backbone,torch.device(a.device))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    print('Prediction written to',a.output)

if __name__=='__main__': main()
