"""이상행동 분류 (ConvLSTM) 학습 스크립트.

source/1. 이상행동 분류/ConvLSTM/tools/train.py 를 정리·독립화.
앱의 core/behavior_classifier.py 가 로드하는 model_final.pth 형식으로 저장한다.

사용법:
    python train.py [--resume output/model_10.pth]

데이터 준비:
    config.py 의 DATA_ROOT, SPLIT_DIR 설정 후
    split/train.csv, split/val.csv 를 준비한다.
    CSV 형식: frame1_path,frame2_path,frame3_path,label  (헤더 없음)
"""

import argparse
import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# ── 같은 폴더의 모듈 import ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from dataset import BehaviorDataset, build_transform, LABEL_NAMES


# ── 모델 정의 (core/behavior_classifier.py 와 동일 구조) ─────────────────────────
class _LSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        _, (h_out, _) = self.lstm(x, (h0, c0))
        return h_out.view(-1, self.hidden_size)


class LSTM_NIA(nn.Module):
    def __init__(self, hidden_size=256, num_classes=8, num_layers=1):
        super().__init__()
        self.input_size = 512  # resnet50 출력 2048 → reshape(-1, 512)
        backbone = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*(list(backbone.children())[:-1]))
        self.lstm = _LSTM(self.input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, img_size = x.shape[0], x.shape[2:]
        x = x.reshape(-1, *img_size)               # (B*seq, C, H, W)
        x = self.backbone(x)                       # (B*seq, 2048, 1, 1)
        x = x.reshape(batch_size, -1, self.input_size)  # (B, seq*4, 512)
        out = self.lstm(x)
        return self.fc(out)


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


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("behavior_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item()


# ── 학습 1 에포크 ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    t0 = time.time()

    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = torch.LongTensor(labels).to(device)

        out = model(imgs)
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        losses.update(loss.item(), bs)
        accs.update(accuracy(out, labels), bs)

        if i % 50 == 0:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch[{epoch}][{i}/{len(loader)}]  "
                f"loss {losses.avg:.4f}  acc {accs.avg*100:.1f}%  "
                f"({elapsed:.0f}s elapsed)"
            )
    return losses.avg, accs.avg


# ── 검증 ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, logger):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    class_correct = [0] * C.NUM_CLASSES
    class_total = [0] * C.NUM_CLASSES

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels_t = torch.LongTensor(labels).to(device)
        out = model(imgs)
        loss = criterion(out, labels_t)
        bs = imgs.size(0)
        losses.update(loss.item(), bs)
        accs.update(accuracy(out, labels_t), bs)

        preds = out.argmax(dim=1).cpu().tolist()
        for p, l in zip(preds, labels):
            class_total[l] += 1
            if p == l:
                class_correct[l] += 1

    logger.info(f"  Val loss {losses.avg:.4f}  acc {accs.avg*100:.1f}%")
    for i, name in enumerate(LABEL_NAMES):
        if class_total[i]:
            logger.info(f"    {name}: {class_correct[i]}/{class_total[i]} "
                        f"({class_correct[i]/class_total[i]*100:.1f}%)")
    return losses.avg, accs.avg


# ── 저장 (앱 호환 포맷) ──────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, output_dir, name):
    """앱의 BehaviorClassifier._load_weights() 가 읽는 포맷으로 저장."""
    state = {
        "epoch": epoch,
        "name": "LSTM_NIA",
        "state_dict": model.state_dict(),  # DataParallel 시 module. 접두사 포함
        "optimizer": optimizer.state_dict(),
    }
    path = os.path.join(output_dir, name)
    torch.save(state, path)
    return path


# ── 진입점 ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="이상행동 분류 학습")
    parser.add_argument("--data-root", default=C.DATA_ROOT)
    parser.add_argument("--split-dir", default=C.SPLIT_DIR)
    parser.add_argument("--output-dir", default=C.OUTPUT_DIR)
    parser.add_argument("--resume", default="", help="재개할 체크포인트 경로")
    parser.add_argument("--epochs", type=int, default=C.END_EPOCH)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    args = parser.parse_args()

    logger = setup_logger(args.output_dir)
    logger.info(f"설정: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # ── 데이터셋 ─────────────────────────────────────────────────────────────────
    transform = build_transform()
    _here = os.path.dirname(os.path.abspath(__file__))
    split_dir = args.split_dir if os.path.isabs(args.split_dir) \
        else os.path.join(_here, args.split_dir)
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(_here, args.data_root)

    train_csv = os.path.join(split_dir, "train.csv")
    val_csv = os.path.join(split_dir, "val.csv")
    assert os.path.isfile(train_csv), f"train.csv 없음: {train_csv}"
    assert os.path.isfile(val_csv), f"val.csv 없음: {val_csv}"

    train_ds = BehaviorDataset(args.data_root, train_csv, transform=transform)
    val_ds = BehaviorDataset(args.data_root, val_csv, transform=transform)
    logger.info(f"train {len(train_ds)}개 / val {len(val_ds)}개")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=C.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=C.NUM_WORKERS, pin_memory=True)

    # ── 모델 ─────────────────────────────────────────────────────────────────────
    model = LSTM_NIA(hidden_size=C.HIDDEN_SIZE, num_classes=C.NUM_CLASSES,
                     num_layers=C.NUM_LAYERS)
    if len(C.GPUS) > 1:
        model = nn.DataParallel(model, device_ids=C.GPUS)
    model = model.to(device)

    # ── 손실함수 ─────────────────────────────────────────────────────────────────
    weight = train_ds.class_weights().to(device) if C.USE_CLASS_WEIGHT else None
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)

    # ── 옵티마이저 / 스케줄러 ────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * C.LR_MIN_RATIO
    )

    begin = C.BEGIN_EPOCH
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        begin = ckpt["epoch"]
        logger.info(f"에포크 {begin} 부터 재개")

    # ── 학습 루프 ─────────────────────────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(begin, args.epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, logger)
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train loss {tr_loss:.4f} acc {tr_acc*100:.1f}%  |  "
            f"val loss {val_loss:.4f} acc {val_acc*100:.1f}%  "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        # 정기 저장
        if (epoch + 1) % C.SAVE_INTERVAL == 0:
            path = save_checkpoint(model, optimizer, epoch + 1,
                                   args.output_dir, f"model_{epoch+1}.pth")
            logger.info(f"저장: {path}")

        # 최고 정확도 모델 별도 저장
        if val_acc > best_acc:
            best_acc = val_acc
            path = save_checkpoint(model, optimizer, epoch + 1,
                                   args.output_dir, "model_best.pth")
            logger.info(f"최고 모델 갱신 ({best_acc*100:.1f}%): {path}")

    # 최종 저장 — 앱이 기본으로 찾는 이름
    path = save_checkpoint(model, optimizer, args.epochs,
                           args.output_dir, "model_final.pth")
    logger.info(f"최종 저장: {path}")


if __name__ == "__main__":
    main()
