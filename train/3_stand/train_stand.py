# -*- coding: utf-8 -*-
"""auto_label.py 로 만든 의사 라벨 데이터셋으로 '물품 가판대' 감지 YOLO 학습.

사전학습 yolo11n.pt 를 fine-tune 한다(단일 클래스 stand).
학습 종료 후 best.pt 를 프로젝트 model/stand.pt 로 배포해 앱에서 바로 사용 가능하게 한다.

사용법:
    python train_stand.py
    python train_stand.py --epochs 100 --imgsz 640 --base yolo11s.pt
"""

import argparse
import os
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))


def main():
    parser = argparse.ArgumentParser(description="가판대 감지 YOLO 학습")
    parser.add_argument("--base", default="yolo11n.pt", help="fine-tune 기반 가중치")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    data_yaml = os.path.join(_HERE, "dataset", "data.yaml")
    assert os.path.isfile(data_yaml), f"data.yaml 없음 — 먼저 auto_label.py 실행: {data_yaml}"

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  base: {args.base}")

    model = YOLO(args.base)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=os.path.join(_HERE, "output"),
        name="stand",
        exist_ok=True,
        patience=20,
        # 가판대는 정적 구조물 — 색/기하 증강은 적당히
        degrees=0.0, shear=0.0, perspective=0.0,
        mosaic=1.0, fliplr=0.5,
        verbose=True,
    )

    best = os.path.join(_HERE, "output", "stand", "weights", "best.pt")
    if os.path.isfile(best):
        dest_dir = os.path.join(_ROOT, "model")
        os.makedirs(dest_dir, exist_ok=True)
        # stand+pos 2클래스 통합 모델 → objects.pt
        dest = os.path.join(dest_dir, "objects.pt")
        shutil.copy(best, dest)
        print(f"\nbest 배포: {best}\n        → {dest}")
    else:
        print(f"\n[경고] best.pt 미생성: {best}")


if __name__ == "__main__":
    main()
