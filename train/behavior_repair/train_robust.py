"""포즈 입력이 섞인 8개 클래스 관절 행동 모델을 학습한다. 관절 증강과 클래스 가중치를 적용하고 검증 결과로 체크포인트를 고른다. 시험과
진단 영상은 학습에 포함하지 않는다.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from diagnose import OUT, LABELS, dump
from config import PREPARED, PREVIOUS
from core.kpt_behavior_classifier import KeypointLSTMv2, TrackBehaviorBuffer
from compare import scores


def augment(x):
    x = x.reshape(-1, 30, 17, 2).clone()
    b = len(x)
    valid = x.ne(0).any(-1, keepdim=True)
    flip = torch.rand(b, device=x.device) < .5
    pairs = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]
    mirrored = x[:, :, pairs].clone()
    mirrored[..., 0] = 1-mirrored[..., 0]
    x[flip] = mirrored[flip]
    valid[flip] = valid[flip][:, :, pairs]
    # 시퀀스 전체에 같은 경계 상자 변형을 적용해 동작의 연속성을 유지한다.
    scale = 1 + torch.randn(b,1,1,2,device=x.device).clamp(-2,2)*.035
    shift = torch.randn(b,1,1,2,device=x.device).clamp(-2,2)*.02
    x = (x-.5)*scale + .5 + shift
    x += torch.randn_like(x)*.006
    # 가려진 관절은 0으로 유지해 보이지 않는 얼굴이나 손 위치를 추가하지 않는다.
    joints = torch.rand(b,1,17,1,device=x.device) >= .04
    frames = torch.ones(b,30,1,1,device=x.device,dtype=torch.bool)
    drop = torch.rand(b,device=x.device) < .25
    at = torch.randint(1,29,(b,),device=x.device)
    frame_ids = torch.arange(30,device=x.device)[None,:]
    frames[:, :, 0, 0] = ~(drop[:,None] & (frame_ids >= at[:,None]) & (frame_ids < at[:,None]+2))
    return (x*valid*joints*frames).reshape(-1,30,34)


def load(path):
    with np.load(path, allow_pickle=False) as d:
        x, y, groups = d['x'], d['y'], set(d['groups'])
    assert x.shape == (len(y),30,34) and np.isfinite(x).all()
    return torch.tensor(x,device='cuda'), torch.tensor(y,device='cuda',dtype=torch.long), groups


@torch.inference_mode()
def evaluate(model,x,y):
    model.eval()
    pred = torch.cat([model(batch).argmax(1) for batch in x.split(256)]).cpu().numpy()
    truth = y.cpu().numpy()
    result = scores(truth,pred)
    result['non_smoking_to_smoking_rate'] = float(np.mean(pred[truth != 4] == 4))
    result['normal_to_anomaly_rate'] = float(np.mean(pred[truth == 0] != 0))
    result['anomaly_to_normal_rate'] = float(np.mean(pred[truth != 0] == 0))
    matrix=np.asarray(result['confusion_matrix'])
    for i,label in enumerate(LABELS):
        support=int(matrix[i].sum())
        fp=int(matrix[:,i].sum()-matrix[i,i])
        fn=int(support-matrix[i,i])
        result['per_class'][label].update({'false_positives':fp,'false_negatives':fn,
            'false_positive_rate':fp/(len(truth)-support) if len(truth)>support else None,
            'false_negative_rate':fn/support if support else None})
    return result


def selection(metric):
    # 모델 선택에는 정상을 포함한 8개 클래스에 같은 가중치를 적용한다.
    return metric['macro_f1']


def checkpoint_equal(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(checkpoint_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return len(left)==len(right) and all(checkpoint_equal(a,b) for a,b in zip(left,right))
    return left==right


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=OUT/'mixed-dataset')
    p.add_argument('--output',type=Path,default=OUT/'trained')
    p.add_argument('--epochs',type=int,default=80)
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    assert torch.cuda.is_available()
    recipe={'version':'20261115-all-class-macro-f1','seed':42,
            'training_sha256':hashlib.sha256((a.dataset/'train.npz').read_bytes()).hexdigest(),
            'validation_s_sha256':hashlib.sha256((a.dataset/'val-s.npz').read_bytes()).hexdigest(),
            'validation_n_sha256':hashlib.sha256((PREPARED/'val.npz').read_bytes()).hexdigest()}
    a.output.mkdir(exist_ok=True)
    job=a.output/'job.json'
    if job.exists():
        if json.loads(job.read_text('utf-8'))!=recipe:
            raise ValueError('Dataset or training recipe changed; use a different output directory.')
    elif any(a.output.iterdir()):
        raise ValueError('Output directory contains an unrecognized training run.')
    dump(job,recipe)
    x,y,groups=load(a.dataset/'train.npz')
    val={}
    for name,path in [('n',PREPARED/'val.npz'),('s',a.dataset/'val-s.npz')]:
        vx,vy,vg=load(path)
        assert not groups & vg, 'Training/validation filming group leakage'
        val[name]=(vx,vy)
    old=TrackBehaviorBuffer._build_and_load(str(PREVIOUS)).cuda()
    baseline={k:evaluate(old,*v) for k,v in val.items()}
    dump(a.output/'baseline-validation.json',baseline)
    del old
    model=KeypointLSTMv2().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=4,factor=.5)
    counts=torch.bincount(y,minlength=8)
    weights=(counts.sum()/(8*counts)).sqrt()
    criterion=nn.CrossEntropyLoss(weight=weights,label_smoothing=.03)
    loader=DataLoader(TensorDataset(x,y),batch_size=128,shuffle=True)
    best=-1
    best_epoch=0
    history=[]
    resume_path=a.output/'resume.pt'
    restored_epoch=None
    if resume_path.exists():
        resume=torch.load(resume_path,map_location='cpu',weights_only=True)
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        best,best_epoch,history=resume['best'],resume['best_epoch'],resume['history']
        torch.set_rng_state(resume['torch_rng'])
        torch.cuda.set_rng_state_all(resume['cuda_rng'])
        assert checkpoint_equal(model.state_dict(),resume['model'])
        assert checkpoint_equal(optimizer.state_dict(),resume['optimizer'])
        assert checkpoint_equal(scheduler.state_dict(),resume['scheduler'])
        assert torch.equal(torch.get_rng_state(),resume['torch_rng'])
        assert checkpoint_equal(torch.cuda.get_rng_state_all(),resume['cuda_rng'])
        restored_epoch=len(history)
        print('Resuming after epoch',len(history),flush=True)
    started=time.time()
    for epoch in range(len(history)+1,a.epochs+1):
        model.train()
        total=0
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(augment(bx)),by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            total+=float(loss.detach())*len(by)
        metrics={k:evaluate(model,*v) for k,v in val.items()}
        score=float(np.mean([selection(v) for v in metrics.values()]))
        scheduler.step(score)
        row={'epoch':epoch,'loss':total/len(y),'selection_score':score,
             'n_macro_f1':metrics['n']['macro_f1'],'s_macro_f1':metrics['s']['macro_f1'],
             'n_per_class_f1':{k:v['f1'] for k,v in metrics['n']['per_class'].items()},
             's_per_class_f1':{k:v['f1'] for k,v in metrics['s']['per_class'].items()}}
        history.append(row)
        dump(a.output/'training-history.json',history)
        print(json.dumps(row),flush=True)
        dump(OUT/'progress.json',{'stage':'training',**row,'elapsed_seconds':time.time()-started})
        if score>best:
            best,best_epoch=score,epoch
            temp=a.output/'best.tmp'
            torch.save({k:v.cpu() for k,v in model.state_dict().items()},temp)
            temp.replace(a.output/'kpt_behavior_repaired.pth')
            dump(a.output/'best-validation.json',metrics)
        temp=resume_path.with_suffix('.tmp')
        torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items()},
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'best':best,'best_epoch':best_epoch,'history':history,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp)
        temp.replace(resume_path)
        if epoch-best_epoch>=12:
            break
    checkpoint=a.output/'kpt_behavior_repaired.pth'
    dump(a.output/'training.json',{'initialization':'random','seed':42,'device':torch.cuda.get_device_name(0),
         'best_epoch':best_epoch,'best_score':best,'history':history,
         'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
         'selection':'equal n/s validation weight; macro F1 gives equal weight to all eight classes including normal',
         'test_used_for_selection':False,'diagnostic_used_for_training':False,
         'resume_verified_epoch':restored_epoch,'bitwise_continuous_resume_guaranteed':False,
         'dataset':str(a.dataset),'class_weights':weights.tolist(),
         'augmentations':'COCO joint-aware mirror, bbox scale/translation, coordinate noise, joint/frame masking',
         'labels':LABELS,'source_fps':3,'sequence_length':30,'detection_interval':1,
         'pose_models':['yolo11n-pose.pt','yolo11s-pose.pt'],'resolution':480,'detection_confidence':.4})
    print('Training completed',checkpoint,flush=True)
    if sys.platform == 'win32':
        # Windows에서 지역 CUDA 텐서를 정리할 때 관측된 메모리 할당기 오류를 피한다.
        # 학습 산출물 저장을 마친 경우에만 이 위치에서 종료한다.
        import ctypes
        torch.cuda.synchronize()
        sys.stdout.flush()
        sys.stderr.flush()
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.TerminateProcess.argtypes=(ctypes.c_void_p,ctypes.c_uint)
        if not kernel.TerminateProcess(kernel.GetCurrentProcess(),0):
            raise ctypes.WinError()


if __name__=='__main__':
    main()
