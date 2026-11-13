"""키포인트 기반 이상행동 분류 (KeypointLSTM) 학습 스크립트.

사용법:
    # 1. 먼저 키포인트 추출
    python extract_keypoints.py --data-root C:/data/NIA --split train
    python extract_keypoints.py --data-root C:/data/NIA --split val

    # 2. 학습
    python train_kpt.py

    # 3. 재개
    python train_kpt.py --resume output/kpt_epoch_10.pth

출력:
    output/kpt_behavior.pth  — 앱이 바로 로드할 수 있는 plain state_dict
"""

import argparse
import os
import sys
import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model_kpt import KeypointLSTMv2
from dataset_kpt import KptBehaviorDataset, LABEL_NAMES

# ── 설정 ─────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(_HERE, "output")
EPOCHS = 80
BATCH_SIZE = 64
LR = 3e-4
LR_MIN = 3e-5
NUM_WORKERS = 0
GPUS = [0]
SAVE_INTERVAL = 10

# 모델 하이퍼파라미터
HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.5   # 과적합 완화 (0.3 → 0.5)


# ── 유틸 ─────────────────────────────────────────────────────────────────────────
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("kpt_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(output_dir, "kpt_train.log"), encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def accuracy(out, target):
    return (out.argmax(1) == target).float().mean().item()


# ── 학습 1 에포크 ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger):
    model.train()
    losses, accs = AverageMeter(), AverageMeter()
    t0 = time.time()

    for i, (seqs, labels) in enumerate(loader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        out = model(seqs)
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = seqs.size(0)
        losses.update(loss.item(), bs)
        accs.update(accuracy(out, labels), bs)

        if i % 50 == 0:
            logger.info(f"[{epoch}][{i}/{len(loader)}] "
                        f"loss {losses.avg:.4f}  acc {accs.avg*100:.1f}%  "
                        f"({time.time()-t0:.0f}s)")
    return losses.avg, accs.avg


# ── 검증 ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, logger):
    model.eval()
    losses, accs = AverageMeter(), AverageMeter()
    class_c = [0] * len(LABEL_NAMES)
    class_t = [0] * len(LABEL_NAMES)

    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        out = model(seqs)
        loss = criterion(out, labels)
        bs = seqs.size(0)
        losses.update(loss.item(), bs)
        accs.update(accuracy(out, labels), bs)

        preds = out.argmax(1).cpu().tolist()
        for p, l in zip(preds, labels.cpu().tolist()):
            class_t[l] += 1
            if p == l:
                class_c[l] += 1

    logger.info(f"  Val loss {losses.avg:.4f}  acc {accs.avg*100:.1f}%")
    for i, name in enumerate(LABEL_NAMES):
        if class_t[i]:
            logger.info(f"    {name}: {class_c[i]}/{class_t[i]} "
                        f"({class_c[i]/class_t[i]*100:.1f}%)")
    return losses.avg, accs.avg


# ── 진입점 ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", default="split/kpt_train.npz")
    parser.add_argument("--val-npz",   default="split/kpt_val.npz")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--no-deploy", action="store_true",
                        help="kpt_behavior.pth 로 배포하지 않음 (실험용)")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    logger = setup_logger(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}  설정: {vars(args)}")

    # npz 경로 해석 (절대 or 스크립트 기준 상대)
    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(_HERE, p)

    train_npz, val_npz = _abs(args.train_npz), _abs(args.val_npz)
    assert os.path.isfile(train_npz), \
        f"train npz 없음: {train_npz}\n  먼저 extract_keypoints.py 를 실행하세요."
    assert os.path.isfile(val_npz), \
        f"val npz 없음: {val_npz}\n  먼저 extract_keypoints.py --split val 을 실행하세요."

    train_ds = KptBehaviorDataset(train_npz, augment=True)
    val_ds   = KptBehaviorDataset(val_npz,   augment=False)
    logger.info(f"train {len(train_ds)}개 / val {len(val_ds)}개")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = KeypointLSTMv2(hidden=args.hidden, num_layers=NUM_LAYERS,
                           num_classes=len(LABEL_NAMES), dropout=args.dropout)
    if len(GPUS) > 1:
        model = nn.DataParallel(model, device_ids=GPUS)
    model = model.to(device)

    weight = train_ds.class_weights().to(device)
    # label_smoothing: 영상 전체에 단일 라벨을 부여한 약지도 노이즈 완화
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=LR_MIN
    )

    begin = 0
    if args.resume and os.path.isfile(args.resume):
        target = model.module if isinstance(model, nn.DataParallel) else model
        target.load_state_dict(torch.load(args.resume, map_location=device,
                                          weights_only=True))
        logger.info(f"재개: {args.resume}")

    best_acc = 0.0
    for epoch in range(begin, args.epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger)
        val_loss, val_acc = validate(model, val_loader, criterion, device, logger)
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{args.epochs}  "
                    f"train {tr_loss:.4f}/{tr_acc*100:.1f}%  "
                    f"val {val_loss:.4f}/{val_acc*100:.1f}%  "
                    f"lr {scheduler.get_last_lr()[0]:.2e}")

        raw = model.module if isinstance(model, nn.DataParallel) else model

        if (epoch + 1) % SAVE_INTERVAL == 0:
            p = os.path.join(args.output_dir, f"kpt_epoch_{epoch+1}.pth")
            torch.save(raw.state_dict(), p)
            logger.info(f"저장: {p}")

        if val_acc > best_acc:
            best_acc = val_acc
            p = os.path.join(args.output_dir, "kpt_best.pth")
            torch.save(raw.state_dict(), p)
            logger.info(f"최고 acc 갱신 ({best_acc*100:.1f}%): {p}")

    # 앱이 기본으로 찾는 이름(kpt_behavior.pth)에는 **best 체크포인트**를 배포한다.
    # 마지막 에포크가 과적합으로 best 보다 나쁠 수 있으므로 best 를 우선한다.
    if args.no_deploy:
        logger.info(f"실험 모드(--no-deploy): 최고 val acc {best_acc*100:.2f}% "
                    f"(배포 생략, kpt_best.pth 만 저장)")
        return

    final_path = os.path.join(args.output_dir, "kpt_behavior.pth")
    best_path = os.path.join(args.output_dir, "kpt_best.pth")
    raw = model.module if isinstance(model, nn.DataParallel) else model
    if os.path.isfile(best_path):
        import shutil
        shutil.copyfile(best_path, final_path)
        logger.info(f"best({best_acc*100:.1f}%) → 배포: {final_path}")
    else:
        torch.save(raw.state_dict(), final_path)
        logger.info(f"최종 저장: {final_path}")


if __name__ == "__main__":
    main()
