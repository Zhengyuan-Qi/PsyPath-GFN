"""Predeclared fixed all-view ablation of the archived hierarchical policy."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_alternating_psychometric_gfn_cached as formal
from scripts.meter_gfn import MODALITY_BUNDLES

def fixed_bundle(module, inputs, output):
    index = int(torch.nonzero(MODALITY_BUNDLES.all(1),as_tuple=False).item())
    logits = torch.full_like(output, -torch.inf)
    logits[...,index] = 0.
    return logits

original_new_model=formal.new_model
def new_model(*args,**kwargs):
    model=original_new_model(*args,**kwargs)
    model.modality_head.register_forward_hook(fixed_bundle)
    return model
formal.new_model=new_model
if __name__=='__main__':
    raise SystemExit(formal.main())
