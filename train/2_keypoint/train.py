"""UniPose 키포인트 학습 스크립트.

source/2. 키포인트 객체 인식/UniPose/tools/train.py 를 정리·독립화.
앱이 로드하는 final_state.pth 형식(plain state_dict, 'module.' 접두사 없음)으로 저장.

사용법:
    python train.py [--resume output/epoch_5.pth]

데이터 준비:
    config.py 의 DATA_ROOT, SPLIT_DIR 설정 후
    split/train.json, split/val.json 을 준비한다.
    JSON 형식: dataset.py 상단 docstring 참조.
"""

import argparse
import os
import sys
import time
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from dataset import KeypointDataset, build_transform
from model import UniPose


# ── 손실: 가중 MSE (JointsMSELoss) ───────────────────────────────────────────────
class JointsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, pred, target, weight):
        batch, J, H, W = pred.shape
        # weight: (B, J, 1) — 0이면 불가시 관절, 손실에서 제외
        pred = pred * weight.unsqueeze(-1)
        target = target * weight.unsqueeze(-1)
        return self.mse(pred, target)


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
    logger = logging.getLogger("keypoint_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"), encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def save_checkpoint(model, optimizer, epoch, output_dir, name):
    """앱 호환 포맷: plain state_dict (module. 접두사 없음).
    DataParallel 사용 시 model.module.state_dict() 로 접두사 제거.
    """
    raw = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(raw.state_dict(), os.path.join(output_dir, name))


# ── 학습 1 에포크 ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger):
    model.train()
    losses = AverageMeter()
    t0 = time.time()

    for i, (imgs, heatmaps, weights) in enumerate(loader):
        imgs = imgs.to(device)
        heatmaps = heatmaps.to(device)
        weights = weights.to(device)

        pred = model(imgs)
        loss = criterion(pred, heatmaps, weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), imgs.size(0))

        if i % 50 == 0:
            logger.info(f"Epoch[{epoch}][{i}/{len(loader)}]  "
                        f"loss {losses.avg:.6f}  ({time.time()-t0:.0f}s)")
    return losses.avg


# ── 검증 ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device, logger):
    model.eval()
    losses = AverageMeter()

    for imgs, heatmaps, weights in loader:
        imgs = imgs.to(device)
        heatmaps = heatmaps.to(device)
        weights = weights.to(device)
        pred = model(imgs)
        loss = criterion(pred, heatmaps, weights)
        losses.update(loss.item(), imgs.size(0))

    logger.info(f"  Val loss {losses.avg:.6f}")
    return losses.avg


# ── 진입점 ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UniPose 키포인트 학습")
    parser.add_argument("--data-root", default=C.DATA_ROOT)
    parser.add_argument("--split-dir", default=C.SPLIT_DIR)
    parser.add_argument("--output-dir", default=C.OUTPUT_DIR)
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=C.END_EPOCH)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    args = parser.parse_args()

    logger = setup_logger(args.output_dir)
    logger.info(f"설정: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # ── 데이터셋 ─────────────────────────────────────────────────────────────────
    split_dir = args.split_dir if os.path.isabs(args.split_dir) \
        else os.path.join(os.path.dirname(__file__), args.split_dir)
    train_json = os.path.join(split_dir, "train.json")
    val_json = os.path.join(split_dir, "val.json")
    assert os.path.isfile(train_json), f"train.json 없음: {train_json}"
    assert os.path.isfile(val_json),   f"val.json 없음: {val_json}"

    train_ds = KeypointDataset(args.data_root, train_json,
                               is_train=True, transform=build_transform(True))
    val_ds = KeypointDataset(args.data_root, val_json,
                             is_train=False, transform=build_transform(False))
    logger.info(f"train {len(train_ds)}개 / val {len(val_ds)}개")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=C.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=C.NUM_WORKERS, pin_memory=True)

    # ── 모델 ─────────────────────────────────────────────────────────────────────
    pretrained = C.BACKBONE_PRETRAINED if os.path.isfile(C.BACKBONE_PRETRAINED) else None
    if pretrained is None:
        logger.warning(f"backbone 가중치 없음: {C.BACKBONE_PRETRAINED}  (랜덤 초기화)")

    model = UniPose(num_joints=C.NUM_JOINTS, backbone=C.BACKBONE,
                    output_stride=C.OUTPUT_STRIDE, pretrained_path=pretrained)
    if len(C.GPUS) > 1:
        model = nn.DataParallel(model, device_ids=C.GPUS)
    model = model.to(device)

    # ── 손실 / 옵티마이저 / 스케줄러 ────────────────────────────────────────────
    criterion = JointsMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=C.LR_STEP, gamma=0.25
    )

    begin = C.BEGIN_EPOCH
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        # resume 저장 포맷: plain state_dict
        target = model.module if isinstance(model, nn.DataParallel) else model
        target.load_state_dict(ckpt)
        logger.info(f"재개: {args.resume}")

    # ── 학습 루프 ─────────────────────────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(begin, args.epochs):
        tr_loss = train_one_epoch(model, train_loader, criterion,
                                  optimizer, device, epoch, logger)
        val_loss = validate(model, val_loader, criterion, device, logger)
        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{args.epochs}  "
                    f"train {tr_loss:.6f}  val {val_loss:.6f}  "
                    f"lr {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % C.SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch + 1,
                            args.output_dir, f"epoch_{epoch+1}.pth")
            logger.info(f"저장: epoch_{epoch+1}.pth")

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1,
                            args.output_dir, "model_best.pth")
            logger.info(f"최저 val loss 갱신 ({best_loss:.6f}): model_best.pth")

    # 앱이 기본으로 찾는 이름으로 최종 저장
    save_checkpoint(model, optimizer, args.epochs,
                    args.output_dir, "final_state.pth")
    logger.info(f"최종 저장: final_state.pth")


if __name__ == "__main__":
    main()
