"""Label-free online inference using the paper's frozen encoder and locked model."""
import gc
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from scripts import run_meter_mh_cache as encoding_contract
from scripts import validate_alternating_psychometric_gfn_cached as formal
from scripts.audit_psypath_reward_flow_faithfulness import checkpoint_args
from scripts.alternating_psychometric_gfn import evidence_features
from deployment import verify_manifest

engine = encoding_contract.engine
VIEWS = ('clinical','answer','context','visual')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def read_input(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    allowed={'dataset','dataset_root','participant_id','task'}
    if set(d)!=allowed: raise ValueError('Input must contain only dataset, dataset_root, participant_id, task; no labels or scores.')
    if d['task'] not in formal.TASKS or formal.TASKS[d['task']]['dataset']!=d['dataset']:
        raise ValueError('Dataset/task mismatch')
    if not str(d['participant_id']).isdigit(): raise ValueError('Expected a release participant ID')
    root=Path(d['dataset_root']).expanduser()
    if not root.is_absolute(): root=Path(path).resolve().parent/root
    if not root.is_dir(): raise FileNotFoundError(root)
    return dict(d,dataset_root=root.resolve(),participant_id=str(d['participant_id']))

@torch.inference_mode()
def encode_interview(spec,backbone,device,progress=print):
    """Read a full release-format interview; no feature cache or label files are read."""
    dataset,root,pid=spec['dataset'],spec['dataset_root'],spec['participant_id']
    audio=engine.base.source_file(dataset,root,pid,'AUDIO.wav')
    turns=engine.base.read_turns(dataset,root,pid)
    if not audio.is_file() or len(turns)<2: raise ValueError('Audio or complete response events missing')
    visual,audit=engine.base.visual_event_features(dataset,root,pid,turns)
    visual=engine.base.finite(visual)
    if visual.ndim!=2 or visual.shape[0]!=len(turns) or visual.shape[1]>engine.MODEL_VISUAL_SIZE:
        raise ValueError('Visual feature contract mismatch')
    visual=np.pad(visual,((0,0),(0,engine.MODEL_VISUAL_SIZE-visual.shape[1])))
    processor,model=engine.load_model_safe(Path(backbone),device)
    vectors={v:[] for v in VIEWS[:3]}; metadata=[]
    try:
        with tempfile.TemporaryDirectory(prefix='psypath_audio_') as temporary:
            for i,turn in enumerate(turns):
                clip=engine.base.write_clip(audio,turn['start'],turn['stop'],Path(temporary)/f'event_{i}.wav',.4,12.)
                history=engine.history_text(turns,i)
                for view in vectors:
                    messages=engine._clinical_messages(dataset,turn,history,clip,view)
                    vectors[view].append(engine.encode_batch_low_memory(processor,model,[messages],device)[0])
                metadata.append(engine._turn_meta(turn,i,bool(history)))
                progress(f'Encoded event {i+1}/{len(turns)}',flush=True)
    finally:
        del model,processor
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    raw={v:engine.base.finite(np.stack(x)) for v,x in vectors.items()}
    raw['visual']=visual
    return raw,metadata

def predict_views(raw,turns,pid,checkpoint,task):
    """Shared numerical path; deliberately accepts neither labels nor true scores."""
    args=checkpoint_args(checkpoint)
    projected={}
    for name in VIEWS:
        p=checkpoint['event_projections']
        projection=p['visual'] if name=='visual' else p['shared_by_view'][name] if p.get('shared_by_view') is not None else p['shared']
        projected[name]=formal.gate.transform(raw[name],projection)
    tensors={k:torch.as_tensor(v,dtype=torch.float32) for k,v in projected.items()}
    temporal=torch.as_tensor(formal.temporal_features(turns),dtype=torch.float32)
    record={'pid':str(pid),'projected_np':projected,'events':formal.joint_event_features(tensors,temporal)}
    model=formal.new_model(record,args,torch.device('cpu'),task_name=task)
    model.load_state_dict(checkpoint['gflownet_state'],strict=True);model.eval()
    seed=formal.stable_seed(formal.stable_seed(task,int(checkpoint['seed']),'test'),'posterior')
    masks=[]
    with torch.inference_mode():
        enc=model.encode(record['events'])
        for j in range(48):
            trajectory=formal.sample_trajectory(model,record,args,formal.stable_seed(seed,str(pid),j),gradient=False,encoding=enc)
            masks.append(trajectory.mask.cpu().numpy().astype(bool))
        features=np.stack([evidence_features(projected,m) for m in masks])
        values=checkpoint['psychometric_operator'].predict(features)
    probability=np.asarray(values['probability']); severity=np.asarray(values['severity'])
    if not np.isfinite(probability).all() or not np.isfinite(severity).all(): raise ValueError('Nonfinite predictions')
    selections=[{'path':j+1,'events':[{'event_index':int(i),'views':[VIEWS[v] for v in np.flatnonzero(m[i])]} for i in np.flatnonzero(m.any(1))]} for j,m in enumerate(masks)]
    threshold=float(checkpoint['threshold'])
    return {'task':task,'probability':float(probability.mean()),'screen_positive':bool(probability.mean()>=threshold),
            'severity':float(severity.mean()),'path_probability_sd':float(probability.std()),'decision_threshold':threshold,
            'paths':48,'training_seed':int(checkpoint['seed']),'event_count':len(turns),'evidence':selections,
            'scope':'Screening estimate, not a clinical diagnosis; path variation is not calibrated clinical uncertainty.'}

def run(spec,model_dir,backbone,device):
    weight,entry=verify_manifest(model_dir,backbone,spec['task'])
    checkpoint=formal.load_trusted_checkpoint(weight,torch.device('cpu'))
    if checkpoint['task']!=spec['task'] or checkpoint['variant']!='no_ensr': raise ValueError('Wrong task/model variant')
    raw,turns=encode_interview(spec,backbone,device)
    result=predict_views(raw,turns,spec['participant_id'],checkpoint,spec['task'])
    result['checkpoint_sha256']=entry['sha256']
    result['inference_mode']='fresh interview encoding; no precomputed participant representations'
    return result
