"""선택된 행동 체크포인트를 분리해 둔 시험 자료에서 평가한다. 클래스별 예측과 정상·이상 혼동, 시간 구간별 결과를 함께 집계한다.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
import torch
from diagnose import (OUT, ROOT, STAGING, LABELS, dump, get_groups,
                      cached_extract, HumanDetector, HumanTracker, labeled_windows,
                      replay, summarize, CASE)
from config import PREPARED, PREVIOUS, LEGACY_V1, LEGACY_V2, POSE_S
from train_robust import evaluate
from core.kpt_behavior_classifier import TrackBehaviorBuffer
from prepare_mixed import select


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,default=OUT/'trained/kpt_behavior_repaired.pth')
    p.add_argument('--output',type=Path,default=OUT/'evaluation.json')
    p.add_argument('--raw-test-per-class',type=int,default=16)
    a=p.parse_args()
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    paths={'old_v1':LEGACY_V1,
           'old_v2':LEGACY_V2,
           'previous_retrained':PREVIOUS,
           'repaired':a.model}
    paths={name:path for name,path in paths.items() if path is not None and path.is_file()}
    models={name:TrackBehaviorBuffer._build_and_load(str(path)).cuda().eval() for name,path in paths.items()}
    result={'checkpoint_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
            'labels':LABELS,'note':'Test clips were not used for model selection; diagnostic clip was previously examined.'}
    with np.load(OUT/'mixed-dataset/train.npz',allow_pickle=False) as d:
        training_groups=set(d['groups'])
    with np.load(OUT/'mixed-dataset/val.npz',allow_pickle=False) as d:
        validation_groups=set(d['groups'])
    for split in ['test','diagnostic']:
        with np.load(PREPARED/f'{split}.npz',allow_pickle=False) as d:
            assert not set(d['groups']) & (training_groups | validation_groups), 'Evaluation filming group leakage'
            x=torch.tensor(d['x'],device='cuda')
            y=torch.tensor(d['y'],device='cuda',dtype=torch.long)
            result[split]={'windows':len(y),'clips':len(set(d['clips'])),'groups':len(set(d['groups'])),
                          'models':{k:evaluate(m,x,y) for k,m in models.items()}}
        print(split,{k:round(v['macro_f1'],4) for k,v in result[split]['models'].items()},flush=True)
        dump(a.output,result)
    if not a.raw_test_per_class:
        return
    _,groups=get_groups()
    detector=HumanDetector(str(POSE_S),.4,imgsz=480)
    embedder=HumanTracker().tracker.embedder
    clips=select(groups['test'],a.raw_test_per_class)
    assert not {c.group for c in clips} & (training_groups | validation_groups), 'Raw test group leakage'
    xs,ys=[],[]
    streams=[]
    for i,clip in enumerate(clips,1):
        try:
            tracks,expected,events,_=cached_extract(clip,detector,lambda:HumanTracker(embedder=embedder),'yolo11s-pose.pt')
            windows,unmatched=labeled_windows(tracks,events,clip.label)
        except (ValueError, TypeError) as error:
            print('Excluded invalid annotation',clip.name,str(error),flush=True)
            streams.append({'clip':clip.name,'group':clip.group,'label':LABELS[clip.label],
                            'windows':0,'error':str(error),'models':{}})
            continue
        if windows:
            xs.extend(r[1] for r in windows)
            ys.extend(r[2] for r in windows)
        # 실행 코드의 지수 이동 평균을 적용한다. 낮은 점수에 임의로 절도 라벨을 부여하지 않는다.
        record={'clip':clip.name,'group':clip.group,'label':LABELS[clip.label],
                'unmatched_events':unmatched,'windows':len(windows),'models':{}}
        for name in ['previous_retrained','repaired']:
            rows=replay(tracks,expected[0],paths[name])
            record['models'][name]=summarize(rows,tracks,events,clip.label)
        streams.append(record)
        dump(OUT/'raw-test-streams.json',streams)
        dump(OUT/'progress.json',{'stage':'independent_raw_video_test','completed':i,'total':len(clips)})
        if i%5==0:
            print('raw test',i,'/',len(clips),flush=True)
    x=torch.tensor(np.stack(xs),device='cuda')
    y=torch.tensor(ys,device='cuda',dtype=torch.long)
    result['s_pose_test']={'attempted_clips':len(clips),'windows':len(ys),
        'clips_with_windows':sum(r['windows']>0 for r in streams),'groups':len(set(r['group'] for r in streams)),
        'models':{k:evaluate(m,x,y) for k,m in models.items()}}
    # 프레임 수에 따른 사건 재현율과 비흡연 주석 구간에서의 흡연 오탐을 집계한다.
    streaming={}
    for name in ['previous_retrained','repaired']:
        per={}
        normal_items=[r['models'][name] for r in streams if name in r['models']]
        normal_classified=sum(r['normal_proxy_classified_frames'] for r in normal_items)
        normal_correct=sum(r['normal_proxy_predictions'].get('normal',0) for r in normal_items)
        per['normal_proxy']={'frames':sum(r['normal_proxy_frames'] for r in normal_items),
            'classified_frames':normal_classified,'false_anomaly_frames':normal_classified-normal_correct,
            'false_alarm_rate_when_classified':1-normal_correct/normal_classified if normal_classified else None}
        for label in LABELS[1:]:
            all_items=[r for r in streams if r['label']==label]
            items=[r['models'][name] for r in all_items if name in r['models']]
            expected=sum(r['event_annotated_frames'] for r in items)
            classified=sum(r['event_classified_frames'] for r in items)
            right=sum(r['event_predictions'].get(label,0) for r in items)
            per[label]={'attempted_clips':len(all_items),'annotation_valid_clips':len(items),
                        'event_frames':expected,'classified_frames':classified,
                        'correct_frames':right,'recall_including_missing':right/expected if expected else None,
                        'wrong_classified_frames':classified-right,
                        'uncertain_frames':sum(r['event_uncertain_frames'] for r in items),
                        'missing_or_collecting_frames':sum(r['event_missing_or_collecting_frames'] for r in items),
                        'visible_wrong_frames':sum(sum(r['event_visible_predictions'].values())
                                                  - r['event_visible_predictions'].get(label,0) for r in items),
                        'normal_in_event_frames':sum(r['event_predictions'].get('normal',0) for r in items),
                        'smoking_frames':sum(r['event_predictions'].get('smoking',0) for r in items),
                        'clips_detected':sum(r['event_predictions'].get(label,0)>0 for r in items)}
        streaming[name]=per
    result['streaming_ema_50_percent']=streaming
    dump(a.output,result)
    print('Evaluation completed',flush=True)


if __name__=='__main__':
    main()
