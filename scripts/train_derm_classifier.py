#!/usr/bin/env python3
"""
Train + export Derm Foundation classifier artifacts for MoleCare.

Requires:
  - Accepted Google HAI-DEF / HF terms for google/derm-foundation
  - HUGGINGFACE_TOKEN
  - Labeled JPEG/PNG folders for melanoma vs benign

Writes:
  <out-dir>/classifier.h5
  <out-dir>/scaler.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from io import BytesIO
from pathlib import Path

import numpy as np


def _load_image_bytes(path: Path) -> bytes:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((448, 448))
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--melanoma-dir", type=Path, required=True)
    parser.add_argument("--benign-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cnn-models/derm-foundation"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-per-class", type=int, default=500)
    args = parser.parse_args()

    if not os.environ.get("HUGGINGFACE_TOKEN"):
        print("ERROR: set HUGGINGFACE_TOKEN before training", file=sys.stderr)
        return 1

    import tensorflow as tf
    from huggingface_hub import from_pretrained_keras, login
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    login(token=os.environ["HUGGINGFACE_TOKEN"])
    print("Loading google/derm-foundation (this can take several minutes)...")
    derm = from_pretrained_keras("google/derm-foundation")
    infer = derm.signatures["serving_default"]

    def embed(png_bytes: bytes) -> np.ndarray:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/encoded": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[png_bytes])
                    )
                }
            )
        ).SerializeToString()
        out = infer(inputs=tf.constant([example]))
        return out["embedding"].numpy().flatten()

    X, y = [], []
    for label, folder in ((0, args.melanoma_dir), (1, args.benign_dir)):
        # label 0 = Melanoma, 1 = NotMelanoma (matches DermFoundationService)
        count = 0
        for path in _iter_images(folder):
            if count >= args.max_per_class:
                break
            try:
                X.append(embed(_load_image_bytes(path)))
                y.append(label)
                count += 1
                if count % 10 == 0:
                    print(f"  class {label}: {count} embeddings")
            except Exception as exc:
                print(f"  skip {path}: {exc}", file=sys.stderr)

    if len(X) < 20:
        print("ERROR: need at least 20 total samples to train a head", file=sys.stderr)
        return 1

    X = np.vstack(X)
    y = np.asarray(y, dtype=np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=args.epochs,
        batch_size=32,
        verbose=1,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = args.out_dir / "classifier.h5"
    scaler_path = args.out_dir / "scaler.pkl"
    model.save(classifier_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Wrote {classifier_path}")
    print(f"Wrote {scaler_path}")
    print("Restart molecare-ml so DermFoundationService picks up the artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
