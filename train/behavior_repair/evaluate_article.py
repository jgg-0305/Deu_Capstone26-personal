"""손 주변 상품 검출과 추적 근거를 평가한다. 상품 주석에 대한 위치 검출 지표와 행동 분류에 결합한 결과를 별도로 정리한다.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import tempfile
import time
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from ultralytics.utils import LOGGER

from aihub_labels import LABELS, index_archives, read_annotation
from config import ROOT
from prepare_aihub import detect_batch, embed_batch, match_actor
from prepare_hand import atomic_json
from train_article import annotations, load_video
sys.path.insert(0,str(ROOT))
from core.article_tracking import ArticleDetector, ArticleSmokingGate, ArticleTracker, applicable_article_evidence, intersection_over_union
from core.detector import HumanDetector
from core.kpt_behavior_classifier import TrackBehaviorBuffer
from core.tracker import HumanTracker


def prepare_pose(cap, count, pose, embedder):
    tracker=HumanTracker(embedder=embedder)
    rows=[]
    for begin in range(0,count,16):
        frames=[]
        for _ in range(begin,min(begin+16,count)):
            ok,frame=cap.read()
            if not ok:
                raise ValueError('video ended before XML')
            frames.append(frame)
        detections=detect_batch(pose,frames)
        embeds=embed_batch(tracker,frames,detections)
        for i,(frame,det,emb) in enumerate(zip(frames,detections,embeds),begin):
            for track in tracker.update(det,frame,embeds=emb):
                if track['observed'] and track['keypoints'] is not None:
                    rows.append((i,track['track_id'],track['bbox'],track['keypoints']))
    return rows


def detector_stats(boxes, candidates):
    available=set(range(len(boxes)))
    hit=0
    for box,_,_ in candidates:
        matches=[(intersection_over_union(box,boxes[j]),j) for j in available]
        if matches:
            score,j=max(matches)
            if score >= .5:
                hit+=1
                available.remove(j)
    return hit


def replay(clip, cap, frames, tracks, events, detector, models):
    truth, disappearances, _, _=annotations(clip)
    article=ArticleTracker()
    for model in models.values():
        model.reset()
    stats=Counter()
    trace=[]
    provenance=[]
    gates={name:ArticleSmokingGate() for name in models}
    smoking_episodes=defaultdict(list)
    active_smoking=defaultdict(dict)
    result=defaultdict(lambda: defaultdict(Counter))
    for start in range(0,frames,12):
        batch=[]
        for _ in range(start,min(start+12,frames)):
            ok,frame=cap.read()
            if not ok:
                raise ValueError('video ended before XML')
            batch.append(frame)
        rows=[tracks.get(i,[]) for i in range(start,start+len(batch))]
        outputs=detector.detect_batch(batch, rows)
        for i,(frame, people, detections) in enumerate(zip(batch,rows,outputs),start):
            if i in truth:
                strong_detections = [item for item in detections if item[1] >= .35]
                stats['annotated_boxes']+=len(truth[i])
                stats['predicted_boxes_on_labeled_frames']+=len(strong_detections)
                stats['matched_boxes_iou50']+=detector_stats(truth[i],strong_detections)
            evidence=article.update(people,detections,i/3,frame=frame)
            seen={t['track_id'] for t in people}
            if evidence:
                trace.append({'frame':i,'tracks': sorted(set(evidence)&seen)})
            for name,model in models.items():
                raw=model.update(people,{t['track_id']:t['keypoints'] for t in people},i/3,frame=frame)
                applicable=applicable_article_evidence(raw,evidence,people)
                deferred=gates[name].update(raw,applicable,people,model.inference_origins,model.abn_conf_thr)
                for mode, hidden in ((name,set()),(name+'_article_previous',applicable),
                                     (name+'_article',applicable | deferred)):
                    active={tid for tid,(label,score) in raw.items() if label==4 and score>=.6 and tid not in hidden}
                    open_episodes=active_smoking[mode]
                    for tid in list(open_episodes):
                        if tid not in active:
                            open_episodes.pop(tid)
                    for tid in active:
                        if tid not in open_episodes:
                            episode={'track':tid,'start':i,'end':i}
                            smoking_episodes[mode].append(episode)
                            open_episodes[tid]=episode
                        else:
                            open_episodes[tid]['end']=i
                for actor,begin,end in events:
                    if begin<=i<=end:
                        prediction=raw.get(actor)
                        visible='missing' if prediction is None else (
                            'uncertain' if model.progress(actor)['state']=='uncertain' else LABELS[prediction[0]])
                        result[name]['event'][visible]+=1
                        resolved='theft' if actor in applicable else 'uncertain' if actor in deferred else visible
                        result[name+'_article']['event'][resolved]+=1
                        if name=='pose' and actor in applicable:
                            stats['rule_actor_event_frames']+=1
                        if clip.name.endswith('DYA_07-29_15-20-29_CE_RGB_DF2_F2') and 134<=i<=143:
                            result[name]['pocket'][visible]+=1
                            result[name+'_article']['pocket'][resolved]+=1
                        if clip.name.endswith('DYA_07-29_15-20-29_CE_RGB_DF2_F2') and 130<=i<=143:
                            provenance.append({'frame':i,'model':name,'result':prediction,
                                               'inference':model.inference_origins.get(actor),'visible':resolved})
                    elif i<begin-6:
                        prediction=raw.get(actor)
                        if prediction is not None:
                            visible='uncertain' if model.progress(actor)['state']=='uncertain' else LABELS[prediction[0]]
                            result[name]['pre_event'][visible]+=1
                            resolved='theft' if actor in applicable else 'uncertain' if actor in deferred else visible
                            result[name+'_article']['pre_event'][resolved]+=1
    stats['flow_bridged_frames']=article.bridged_frames
    stats['weak_bridged_frames'] = article.weak_bridged_frames
    stats['flow_reacquired_tracks']=article.reacquired_tracks
    return {'detector':dict(stats),'article_signal':trace,'xml_disappearances':disappearances,
            'models':result,'smoking_episodes':smoking_episodes,'xml_events':events,'inference_provenance':provenance}


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--hand-run',type=Path,required=True)
    parser.add_argument('--pose-cache',type=Path)
    parser.add_argument('--pose-cache-only',action='store_true')
    parser.add_argument('--wait-for-detector',action='store_true')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    LOGGER.setLevel('ERROR')
    clips,_=index_archives(args.data)
    by_name={c.name:c for c in clips if c.source_split=='Validation'}
    manifest=json.loads((args.hand_run/'manifest.json').read_text('utf-8'))
    test=[r for r in manifest if r['split'] in ('test','diagnostic') and 'error' not in r]
    # 기존 평가 분할을 재현하고 이미 확인한 회귀 영상은 별도로 유지한다.
    test.sort(key=lambda row: (row['split'] != 'diagnostic',row['label']!=6,row['label']))
    folder=args.pose_cache if args.pose_cache is not None else args.out/'test_pose'
    folder.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out) as temp:
        scratch=Path(temp)
        if args.pose_cache_only:
            pose=HumanDetector(str(ROOT/'yolo11n-pose.pt'),.4,imgsz=480)
            embedder=HumanTracker().tracker.embedder
            for i,row in enumerate(test,1):
                clip=by_name[row['name']]
                path=folder/(clip.name+'.npz')
                if not path.exists():
                    with ZipFile(clip.label_zip) as z:
                        expected,_=read_annotation(z,clip)
                    cap=load_video(clip,scratch)
                    try:
                        rows=prepare_pose(cap,expected[0],pose,embedder)
                    finally:
                        cap.release()
                    with path.with_suffix('.tmp').open('wb') as file:
                        np.savez_compressed(file,ids=np.array([(a,b) for a,b,_,_ in rows],dtype=np.int32).reshape(-1,2),
                                            boxes=np.array([c for _,_,c,_ in rows]),
                                            points=np.array([d for _,_,_,d in rows]),frames=expected[0])
                    path.with_suffix('.tmp').replace(path)
                print('pose',i,'/',len(test),clip.name,flush=True)
            return
        weights=args.out/'article.pt'
        until=time.monotonic()+1800
        if args.wait_for_detector:
            while not weights.exists():
                if time.monotonic()>until:
                    raise TimeoutError('Waiting for pose cache or article training')
                time.sleep(4)
        checkpoints={'pose': ROOT/'train/1_behavior/retrained-20261115/kpt_behavior_repaired_3fps.pth',
                     'hand': args.hand_run/'hand_fusion_3fps.pth'}
        models={name:TrackBehaviorBuffer(str(path)) for name,path in checkpoints.items()}
        detector=ArticleDetector(weights)
        dataset=json.loads((args.out/'dataset.json').read_text('utf-8'))
        assert not {r[2] for r in dataset} & {r['group'] for r in test}
        results=[]
        for i,row in enumerate(test,1):
            clip=by_name[row['name']]
            tracks=defaultdict(list)
            positions=defaultdict(dict)
            source=folder/(clip.name+'.npz')
            if args.wait_for_detector:
                while not source.exists():
                    if time.monotonic()>until:
                        raise TimeoutError(f'Waiting for pose cache {clip.name}')
                    time.sleep(2)
            with np.load(source,allow_pickle=False) as d:
                count=int(d['frames'])
                for (index,tid),box,points in zip(d['ids'],d['boxes'],d['points']):
                    index,tid=int(index),int(tid)
                    tracks[index].append({'track_id':tid,'bbox':box,'keypoints':points,'observed':True})
                    positions[tid][index]=(box,points)
            with ZipFile(clip.label_zip) as archive:
                _,markers=read_annotation(archive,clip)
            events=[(match_actor(positions,start),start[0],end[0]) for start,end in markers]
            cap=load_video(clip,scratch)
            try:
                metrics=replay(clip,cap,count,tracks,events,detector,models)
            finally:
                cap.release()
            results.append({**row,**metrics})
            atomic_json(args.out/'evaluation.json',results)
            print('evaluation',i,'/',len(test),clip.name,flush=True)


if __name__=='__main__':
    main()
