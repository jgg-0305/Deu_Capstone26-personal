"""손목 주변의 상품 탐지 모델을 학습한다. 상품 주석 상자를 손 주변의 좌표로 바꾸고, 다른 행동의 손 입력도 포함한 YOLO 학습 자료를
만든다.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import cv2
import numpy as np
from ultralytics import YOLO

from aihub_labels import index_archives, split_clips, remove_cross_split_duplicates
from config import ROOT
from prepare_hand import CASE, atomic_json
sys.path.insert(0, str(ROOT))
from core.article_tracking import wrist_roi


def annotations(clip):
    with ZipFile(clip.label_zip) as archive:
        xml = ET.fromstring(archive.read(clip.xml))
    boxes, outside, poses, actors = defaultdict(list), [], defaultdict(dict), {}
    for track in xml.findall('track'):
        label = track.get('label')
        if label == 'object_article':
            active = False
            for b in track.findall('box'):
                index = int(b.get('frame'))
                coords = tuple(float(b.get(k)) for k in ('xtl','ytl','xbr','ybr'))
                if b.get('outside') == '0':
                    active = True
                    if b.get('occluded') != '1' and coords[2] > coords[0] and coords[3] > coords[1]:
                        boxes[index].append(coords)
                elif active:
                    outside.append(index)
                    active = False
        elif label in ('Left hand','Right hand','Left elbow','Right elbow'):
            for point in track.findall('points'):
                xy = point.get('points','').split(',')
                if point.get('outside') == '0' and len(xy) == 2:
                    poses[int(point.get('frame'))].setdefault(label, []).append(tuple(map(float,xy)))
        elif label.endswith('_start') or label.endswith('_end'):
            for b in track.findall('box'):
                if b.get('outside') == '0':
                    actors[int(b.get('frame'))] = float(b.get('ybr'))-float(b.get('ytl'))
    return boxes, outside, poses, actors


def load_video(clip, directory):
    with ZipFile(clip.video_zip) as archive, archive.open(clip.video) as video:
        with (directory/'clip.mp4').open('wb') as target:
            shutil.copyfileobj(video,target)
    before = Path.cwd()
    os.chdir(directory)
    try:
        cap = cv2.VideoCapture('clip.mp4')
    finally:
        os.chdir(before)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f'Cannot decode {clip.name}')
    return cap


def choose(clips, label, limit):
    groups = set()
    for clip in sorted((c for c in clips if c.label==label),
                       key=lambda c: sha256(('article-v1'+c.name).encode()).digest()):
        if clip.group not in groups and len(groups) < limit:
            groups.add(clip.group)
            yield clip


def prepare(clip, split, output, scratch, size):
    boxes, _, poses, actors = annotations(clip)
    if clip.label == 6:
        usable = sorted(boxes)[::2] if size == 160 else sorted(boxes)
    else:
        usable = sorted(poses)[::5][:18] if size == 160 else sorted(poses)[::3][:36]
    if not usable:
        return 0, 0
    usable = set(usable)
    cap = load_video(clip,scratch)
    positive = negative = 0
    try:
        for i in range(max(usable)+1):
            if not cap.grab():
                break
            if i not in usable:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            h,w = frame.shape[:2]
            scale = actors[min(actors,key=lambda f:abs(i-f))] if actors else h*.35
            crops = []
            for side in ('Left','Right'):
                for wrist in poses[i].get(side+' hand',()):
                    elbows = poses[i].get(side+' elbow',())
                    elbow = min(elbows,key=lambda xy: np.linalg.norm(np.subtract(xy,wrist))) if elbows else None
                    crops.append(wrist_roi(wrist,elbow,scale,w,h))
                    if size > 160 and clip.label == 6:
                        crops.append(wrist_roi(wrist,elbow,scale*.74,w,h))
            for j,(x1,y1,x2,y2) in enumerate(dict.fromkeys(crops)):
                if min(x2-x1,y2-y1)<16:
                    continue
                labels=[]
                for a,b,c,d in boxes.get(i,()):
                    l,t,r,bot=max(a,x1),max(b,y1),min(c,x2),min(d,y2)
                    if r<=l or bot<=t or (r-l)*(bot-t) < .7*(c-a)*(d-b):
                        continue
                    labels.append(f'0 {((l+r)/2-x1)/(x2-x1):.6f} {((t+bot)/2-y1)/(y2-y1):.6f} '
                                  f'{(r-l)/(x2-x1):.6f} {(bot-t)/(y2-y1):.6f}')
                # 절도 영상에 보이는 다른 상품에는 주석이 없을 수 있다.
                if not labels and clip.label==6:
                    continue
                stem=f'{clip.name}_{i}_{j}'
                crop = cv2.resize(frame[y1:y2,x1:x2], (size,size), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(output/'images'/split/(stem+'.jpg')),crop)
                (output/'labels'/split/(stem+'.txt')).write_text('\n'.join(labels),encoding='utf-8')
                positive += bool(labels)
                negative += not labels
    finally:
        cap.release()
    return positive,negative


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--size',type=int,choices=[160,256],default=160)
    parser.add_argument('--epochs',type=int,default=22)
    parser.add_argument('--initialize',type=Path,default=ROOT/'model/objects.pt')
    parser.add_argument('--batch',type=int)
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--run-name',default='detector')
    args=parser.parse_args()
    output=args.out.resolve()
    output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    import torch
    torch.set_num_threads(2)
    paths=output/'dataset.json'
    spec_path=output/'crop-spec.json'
    spec={'source':str(args.data.resolve()),'size':args.size,'crop_policy':'wrist-article-v2'}
    if spec_path.exists() and json.loads(spec_path.read_text('utf-8')) != spec:
        raise ValueError('Use a new output directory for a different source or crop size.')
    if not spec_path.exists():
        atomic_json(spec_path,spec)
    if not paths.exists():
        clips,_=index_archives(args.data)
        groups=split_clips(clips,[CASE])
        remove_cross_split_duplicates(groups)
        report=[]
        with tempfile.TemporaryDirectory(dir=output) as scratch:
            plan = [('train',52,7),('val',12,3)] if args.size == 160 else [('train',72,18),('val',16,5)]
            for split,positive,negative in plan:
                for kind in ('images','labels'):
                    (output/kind/split).mkdir(parents=True,exist_ok=True)
                selected=list(choose(groups[split],6,positive))
                for label in (1,2,3,4,5,7):
                    frequent_negative = label==4 or (label==5 and args.size>160)
                    selected.extend(choose(groups[split],label,negative if frequent_negative else max(1,negative//3)))
                for clip in selected:
                    pos,neg=prepare(clip,split,output,Path(scratch),args.size)
                    report.append([split,clip.name,clip.group,pos,neg])
                    print(split,clip.name,pos,neg,flush=True)
        atomic_json(paths,report)
    yaml=output/'article.yaml'
    yaml.write_text(f'path: "{output.as_posix()}"\ntrain: images/train\nval: images/val\nnames:\n  0: article\n',encoding='utf-8')
    model=YOLO(str(args.initialize.resolve()))
    model.train(data=str(yaml),imgsz=args.size,epochs=args.epochs,
                batch=args.batch or (48 if args.size==160 else 32),device=0,workers=args.workers,
                project=str(output),name=args.run_name,exist_ok=True,seed=42,close_mosaic=4,
                mosaic=.4,mixup=0,plots=False,cache='disk' if args.workers else 'ram',patience=7)
    checkpoint = output/'article.part'
    shutil.copyfile(output/args.run_name/'weights/best.pt',checkpoint)
    checkpoint.replace(output/'article.pt')


if __name__=='__main__':
    main()
