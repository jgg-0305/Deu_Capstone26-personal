"""TensorFlow로 KeypointLSTM v2를 학습하고 별도 검증 결과를 저장한다."""

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers, ops

LABELS = ("normal", "fall", "vandalism", "arson", "smoking", "abandon", "theft", "assault")


@keras.saving.register_keras_serializable(package="cctv")
class MotionFeatures(layers.Layer):
    def call(self, inputs):
        valid = ops.cast(ops.not_equal(inputs, 0), inputs.dtype)
        delta = (inputs[:, 1:] - inputs[:, :-1]) * valid[:, 1:] * valid[:, :-1]
        velocity = ops.concatenate([ops.zeros_like(inputs[:, :1]), delta], axis=1)
        return ops.concatenate([inputs, velocity], axis=-1)


def build_model():
    inputs = keras.Input((30, 34))
    features = MotionFeatures()(inputs)
    features = layers.LayerNormalization(epsilon=1e-5, name="norm")(features)
    features = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="lstm0")(features)
    features = layers.Dropout(.5)(features)
    features = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="lstm1")(features)
    attention = layers.Dense(1, name="attn")(features)
    weights = ops.softmax(attention, axis=1)
    pooled = ops.sum(features * weights, axis=1)
    hidden = layers.Dense(64, activation="relu", name="head1")(layers.Dropout(.5)(pooled))
    return keras.Model(inputs, layers.Dense(8, name="head3")(hidden))


def export_arrays(model, output, samples):
    arrays = {}
    for name in ("norm", "attn", "head1", "head3"):
        weight, bias = model.get_layer(name).get_weights()
        arrays[name + ".weight"] = weight.T
        arrays[name + ".bias"] = bias
    for index in range(2):
        layer = model.get_layer(f"lstm{index}")
        for suffix, direction in (("", layer.forward_layer), ("_reverse", layer.backward_layer)):
            kernel, recurrent, bias = direction.get_weights()
            # 두 구현의 게이트 순서는 같으며, PyTorch의 추가 편향은 0으로 둔다.
            arrays[f"lstm.weight_ih_l{index}{suffix}"] = kernel.T
            arrays[f"lstm.weight_hh_l{index}{suffix}"] = recurrent.T
            arrays[f"lstm.bias_ih_l{index}{suffix}"] = bias
            arrays[f"lstm.bias_hh_l{index}{suffix}"] = np.zeros_like(bias)
    arrays["samples"] = samples
    arrays["expected_logits"] = model(samples, training=False).numpy()
    np.savez_compressed(output, **arrays)


def load_splits(directory):
    data = {}
    seen = set()
    for split in ("train", "val", "test"):
        with np.load(directory / f"{split}.npz", allow_pickle=False) as archive:
            x, y, groups = (archive[key] for key in ("x", "y", "groups"))
        if (x.shape != (len(y), 30, 34) or y.shape != (len(x),)
                or not np.issubdtype(y.dtype, np.integer) or not np.isfinite(x).all()
                or set(y) != set(range(8)) or groups.shape != (len(x),)):
            raise ValueError(f"{split}의 입력, 라벨 또는 그룹이 유효하지 않습니다.")
        if seen.intersection(groups):
            raise ValueError(f"{split}에 이전 분할과 같은 촬영 그룹이 있습니다.")
        seen.update(groups)
        data[split] = (x.astype(np.float32), y)
    return data


def report_metrics(y, pred):
    matrix = np.zeros((8, 8), dtype=int)
    np.add.at(matrix, (y, pred), 1)
    diagonal = matrix.diagonal()
    precision = diagonal / np.maximum(1, matrix.sum(axis=0))
    recall = diagonal / matrix.sum(axis=1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-10)
    return {"accuracy": float(diagonal.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
            "labels": LABELS, "confusion_matrix": matrix.tolist(),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                "f1": float(f1[i]), "support": int(matrix[i].sum())}
                          for i, name in enumerate(LABELS)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("에포크와 배치 크기는 1 이상이어야 합니다.")
    devices = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not devices:
        parser.error("TensorFlow에서 GPU를 찾지 못했습니다.")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    keras.utils.set_random_seed(42)
    metadata = json.loads((args.dataset / "dataset.json").read_text(encoding="utf-8"))
    if tuple(metadata["labels"]) != LABELS:
        parser.error("데이터의 행동 클래스 순서를 확인하세요.")
    data = load_splits(args.dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    model = build_model()
    model.compile(optimizer=keras.optimizers.AdamW(3e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    x, y = data["train"]
    counts = np.bincount(y, minlength=8)
    weights = {i: float(len(y) / (8 * count)) for i, count in enumerate(counts)}
    best = args.output / "behavior.keras"
    history = model.fit(x, y, validation_data=data["val"], epochs=args.epochs,
                        batch_size=args.batch_size, class_weight=weights,
                        callbacks=[keras.callbacks.ModelCheckpoint(best, save_best_only=True),
                                   keras.callbacks.EarlyStopping(patience=8)])
    model = keras.models.load_model(best)
    x_test, y_test = data["test"]
    logits = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    result = {"device": str(devices) if devices else "CPU", "dataset": metadata,
              "metrics": report_metrics(y_test, logits.argmax(axis=1)), "history": history.history}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    export_arrays(model, args.output / "weights.npz", x_test[:8])


if __name__ == "__main__":
    main()
