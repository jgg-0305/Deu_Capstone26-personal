"""TensorFlow에서 학습한 가중치를 웹 첨부용 PyTorch 상태 사전으로 변환한다."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.behavior_upload import inspect_checkpoint
from core.kpt_behavior_classifier import KeypointLSTMv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.suffix != ".pth" or args.output.exists():
        parser.error("기존 파일을 덮어쓰지 않는 새 .pth 경로를 지정하세요.")
    torch.set_num_threads(2)
    model = KeypointLSTMv2().eval()
    with np.load(args.weights, allow_pickle=False) as arrays:
        keys = {}
        for key in model.state_dict():
            source = key.replace("head.1.", "head1.").replace("head.3.", "head3.")
            keys[key] = torch.from_numpy(arrays[source])
        model.load_state_dict(keys)
        with torch.inference_mode():
            logits = model(torch.from_numpy(arrays["samples"])).numpy()
        np.testing.assert_allclose(logits, arrays["expected_logits"], rtol=1e-4, atol=1e-5)
    try:
        with args.output.open("xb") as stream:
            torch.save(model.state_dict(), stream)
    except FileExistsError:
        parser.error("출력 파일이 이미 있습니다.")
    try:
        print(inspect_checkpoint(args.output), args.output)
    except Exception:
        args.output.unlink()
        raise


if __name__ == "__main__":
    main()
