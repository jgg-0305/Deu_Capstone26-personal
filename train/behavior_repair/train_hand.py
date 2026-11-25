"""관절과 손 영상을 함께 사용하는 분류기를 학습한다. 촬영 그룹을 분리하고 같은 검증 영상에서 포즈 기반 모델과 융합 모델을 비교한다.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aihub_labels import LABELS
from compare import scores
from config import ROOT
from prepare_hand import atomic_json
from prepare_aihub import inference_windows
sys.path.insert(0, str(ROOT))
from core.hand_behavior import ARCHITECTURE, HandFusion, INPUT_SIZE
from core.kpt_behavior_classifier import TrackBehaviorBuffer
from backend.behavior_upload import inspect_checkpoint


def load_clip(root, row):
    with np.load(root/row['path'], allow_pickle=False) as data:
        tracks = defaultdict(dict)
        for (tid, frame), features in zip(data['ids'], data['x']):
            if features[:34].any():
                tracks[int(tid)][int(frame)] = (None, features)
        return tracks, data['events'].tolist(), data['detector_ious'].tolist(), int(data['frames'])


def windows(root, rows):
    inputs, labels = [], []
    for row in rows:
        if 'error' in row:
            continue
        tracks, events, _, _ = load_clip(root, row)
        if not events:
            continue
        normal_before = min(event[1] for event in events)-6
        for tid in {event[0] for event in events if event[0] >= 0 and not event[3]}:
            positions = tracks[tid]
            # 실행 코드가 유효한 포즈 창을 고르는 규칙을 재사용한다.
            poses = {f: (box, x[:34]) for f, (box, x) in positions.items()}
            empty = np.zeros(INPUT_SIZE, np.float16)
            intervals = [(start, end) for actor, start, end, conflict in events if actor == tid and not conflict]
            for frame, _ in inference_windows(poses):
                label = row['label'] if any(s <= frame <= e for s, e in intervals) else 0 if frame < normal_before else None
                if label is not None:
                    inputs.append(np.stack([positions[i][1] if i in positions else empty for i in range(frame-29, frame+1)]))
                    labels.append(label)
    return TensorDataset(torch.from_numpy(np.stack(inputs)), torch.tensor(labels))


@torch.inference_mode()
def classify(model, dataset, fusion):
    model.eval()
    actual, predicted = [], []
    for x, y in DataLoader(dataset, batch_size=96):
        x = x.cuda().float()
        predicted.extend(model(x if fusion else x[..., :34]).argmax(1).cpu().tolist())
        actual.extend(y.tolist())
    return scores(np.array(actual), np.array(predicted))


def summarize_stream(root, rows, models):
    totals = {name: defaultdict(Counter) for name in models}
    details = []
    for row in rows:
        if 'error' in row:
            details.append(row)
            continue
        tracks, events, ious, count = load_clip(root, row)
        frames = defaultdict(dict)
        for tid, positions in tracks.items():
            for i, (_, x) in positions.items():
                frames[i][tid] = x
        detail = {**row, 'events': events, 'yolo_marker_ious': ious, 'models': {}}
        for name, model in models.items():
            model.reset()
            counts, pre, pocket, quality = Counter(), Counter(), Counter(), Counter()
            for i in range(count):
                values = {tid: x[:model.feature_size] for tid, x in frames[i].items()}
                observed = [{'track_id': tid, 'bbox': [0, 0, 1, 1]} for tid in values]
                pred = model.update(observed, {}, source_seconds=i/3, features=values)
                for tid, begin, end, _ in events:
                    result = pred.get(tid)
                    if result is None:
                        label = 'no_prediction'
                    elif model.progress(tid)['state'] == 'uncertain':
                        label = 'uncertain'
                    else:
                        label = LABELS[result[0]]
                    if begin <= i <= end:
                        counts[label] += 1
                        if tid < 0:
                            quality['unmatched_xml_actor'] += 1
                        elif tid not in values:
                            quality['actor_unobserved'] += 1
                        else:
                            full = frames[i][tid]
                            visible_wrists = int(full[34] > 0) + int(full[42] > 0)
                            quality[f'visible_wrists_{visible_wrists}'] += 1
                            if result is None:
                                quality['buffer_not_ready'] += 1
                        if result and result[1] >= .6 and result[0]:
                            quality['above_event_threshold_'+LABELS[result[0]]] += 1
                    if i < min(event[1] for event in events)-6:
                        pre[label] += 1
                    if row['split'] == 'diagnostic' and 134 <= i <= 143:
                        pocket[label] += 1
            totals[name][LABELS[row['label']]].update(counts)
            totals[name]['normal_proxy'].update(pre)
            detail['models'][name] = {'event': dict(counts), 'pre_event': dict(pre),
                                       'pocket_frames_134_143': dict(pocket), 'quality': dict(quality)}
        details.append(detail)
    return totals, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=28)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    rows = json.loads((args.out/'manifest.json').read_text('utf-8'))
    splits = {name: [r for r in rows if r['split'] == name]
              for name in ('train', 'val', 'test', 'diagnostic')}
    group_sets = [{r['group'] for r in split} for split in splits.values()]
    if not all(splits.values()) or sum(map(len, group_sets)) != len(set.union(*group_sets)):
        raise ValueError('Missing split or overlapping filming groups')
    models_dir = ROOT/'train/1_behavior'
    previous = models_dir/'retrained-20261115/kpt_behavior_repaired_3fps.pth'
    paths = {'existing_pose': previous, 'original_v2': models_dir/'output/kpt_behavior.pth'}
    model = HandFusion().cuda()
    model.pose.load_state_dict(TrackBehaviorBuffer._build_and_load(previous).state_dict())
    # 추가 입력의 기여만 확인하기 위해 포즈 입력을 고정한다.
    model.pose.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.01)
    train, val = (windows(args.out, splits[name]) for name in ('train', 'val'))
    counts = train.tensors[1].bincount(minlength=8)
    if not torch.all(counts > 0) or len(val.tensors[1].unique()) != 8:
        raise ValueError('Training or validation is missing a behavior class')
    base_validation = classify(model.pose, val, False)['macro_f1']
    print(f'Training windows: {len(train)}, validation windows: {len(val)}, '
          f'pose validation Macro-F1: {base_validation:.4f}', flush=True)
    loss = nn.CrossEntropyLoss(weight=(counts.float().median()/counts.float()).sqrt().cuda(), label_smoothing=.02)
    loader = DataLoader(train, batch_size=48, shuffle=True)
    checkpoint = args.out/'hand_fusion_3fps.pth'
    config = {'architecture': ARCHITECTURE, 'labels': list(LABELS),
              'sequence_length': 30, 'sample_fps': 3., 'pose_model': 'yolo11n-pose.pt',
              'resolution': 480, 'detection_confidence': .4}
    best, best_epoch = -1., 0
    epochs = []
    for epoch in range(1, args.epochs+1):
        model.train()
        model.pose.eval()
        for x, y in loader:
            x, y = x.cuda().float(), y.cuda()
            context = x[..., 34:]
            # 한 종류의 단서에 의존하지 않도록 일부 트랙에서 주변 영상 특징을 제외한다.
            context[torch.rand(len(x), device=x.device) < .1] = 0
            context[..., 16:] *= torch.empty(len(x), 1, 1, device=x.device).uniform_(.8, 1.2)
            optimizer.zero_grad(set_to_none=True)
            loss(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
        metric = classify(model, val, True)['macro_f1']
        epochs.append({'epoch': epoch, 'validation_macro_f1': metric})
        if metric > best:
            best, best_epoch = metric, epoch
            torch.save({'state_dict': model.state_dict(), 'input_config': config}, checkpoint.with_suffix('.tmp'))
            checkpoint.with_suffix('.tmp').replace(checkpoint)
        atomic_json(args.out/'progress.json', {'stage': 'train', 'done': epoch, 'total': args.epochs,
                                               'best_epoch': best_epoch, 'best_macro_f1': best})
        print(epochs[-1], flush=True)
        if epoch-best_epoch >= 7:
            break
    paths['hand_fusion'] = checkpoint
    upload_kind = inspect_checkpoint(checkpoint)
    test = windows(args.out, splits['test'])
    metrics = {}
    for name, path in paths.items():
        network = TrackBehaviorBuffer._build_and_load(path).cuda()
        metrics[name] = classify(network, test, name == 'hand_fusion')
    replay = {name: TrackBehaviorBuffer(str(path)) for name, path in paths.items()}
    totals, details = summarize_stream(args.out, splits['test'], replay)
    _, diagnostic = summarize_stream(args.out, splits['diagnostic'], replay)
    atomic_json(args.out/'evaluation.json', {
        'checkpoint_sha256': {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        'split_clips': {name: len(items) for name, items in splits.items()},
        'split_exclusions': {name: [r for r in items if 'error' in r] for name, items in splits.items()},
        'training_counts': dict(zip(LABELS, counts.tolist())), 'epochs': epochs,
        'existing_pose_validation_macro_f1': base_validation,
        'best_epoch': best_epoch, 'upload_validation': upload_kind,
        'windows': len(test), 'window_metrics': metrics, 'stream_counts': totals,
        'stream_details': details, 'known_case': diagnostic,
    })
    atomic_json(args.out/'progress.json', {'stage': 'completed', 'best_epoch': best_epoch})
    print('Completed', checkpoint, flush=True)


if __name__ == '__main__':
    main()
