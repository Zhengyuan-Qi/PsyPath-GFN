import os, importlib, sys, types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from easydict import EasyDict
UP = Path(os.environ.get('PSYPATH_GSIT_UPSTREAM', str(Path(__file__).resolve().parents[1]/'third_party')))
def package(name, path):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod

def valid_resample(x, lengths):
    # Apply exactly the same deterministic adapter to the three new methods.
    return torch.cat([F.interpolate(x[i:i+1, :int(n)].transpose(1, 2),
                     size=50, mode='linear', align_corners=False).transpose(1, 2)
                     for i, n in enumerate(lengths)], dim=0)

class GsiTCore(nn.Module):
    def __init__(self):
        super().__init__()
        path = UP/'GsiT/src/MMSA-GsiT/models/custom/GSIT/modules'
        package('gsit_modules', path)
        package('gsit_modules.GraphAttentions', path/'GraphAttentions')
        # Select the author's existing dense PyTorch backend, not a new attention design.
        file = path/'GraphAttentions/GraphMultiheadAttention.py'
        src = file.read_text()
        assert src.count('from xformers.ops import fmha') == 1
        assert src.count('naive = False') == 1
        src = src.replace('from xformers.ops import fmha', '# Dense author backend selected')
        src = src.replace('naive = False', 'naive = True')
        name = 'gsit_modules.GraphAttentions.GraphMultiheadAttention'
        mod = types.ModuleType(name)
        mod.__package__ = 'gsit_modules.GraphAttentions'
        sys.modules[name] = mod
        exec(compile(src, str(file), 'exec'), mod.__dict__)
        gf = importlib.import_module('gsit_modules.GraphAttentions.GraphFormer')
        sys.modules['gsit_modules.GraphAttentions'].GraphFormerEncoder = gf.GraphFormerEncoder
        package('gsit_modules.CMG', path/'CMG')
        cmg = importlib.import_module('gsit_modules.CMG.CMG')
        cfg = EasyDict(dst_feature_dim_nheads=[32,4], bidirectional=False,
            nlevels=3, attn_dropout=.1, relu_dropout=.1, res_dropout=.1,
            embed_dropout=.1, attn_mask=False)
        self.graph = cmg.CrossModalGraph(EasyDict(cmg_cfg=cfg))
        self.projections = nn.ModuleList([nn.Conv1d(32,32,1,bias=False) for _ in range(3)])
        self.head = nn.Sequential(nn.Dropout(.1),nn.Linear(192,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self, t, a, v):
        seq = [p(x.transpose(1,2)).permute(2,0,1) for p,x in zip(self.projections,[t,v,a])]
        out = self.graph(torch.cat(seq),[50,50,50],False)
        return self.head(torch.cat([s[:,-1] for s in out.split],dim=-1))

class External(nn.Module):
    def __init__(self, name='gsit'):
        super().__init__()
        if name != 'gsit':
            raise ValueError(name)
        self.name = name
        self.auxiliary = None
        self.core = GsiTCore()
    def forward(self, text, audio, visual, length):
        t,a,v = [valid_resample(x,length) for x in [text,audio,visual]]
        self.auxiliary = t.new_zeros(())
        return self.core(t,a,v)
