"""
train_model.py

Trains a simple TensorFlow text-classification model to detect malicious websites
based on their raw HTML content. Processes websites in batches of 2000, fetching
HTML via curl in parallel, up to a total of 20,000 websites.

Usage:
    python train_model.py --dataset websites_labeled.csv

Optional flags:
    --dataset         Path to normalized CSV (domain, label) — default: websites_labeled.csv
    --total           Total websites to train on              — default: 20000
    --batch-size      Websites per training batch             — default: 2000
    --epochs          Epochs per batch                        — default: 3
    --max-tokens      Vocabulary size for text vectorization  — default: 20000
    --max-len         Max HTML token sequence length          — default: 1000
    --timeout         Curl timeout per site in seconds        — default: 10
    --workers         Concurrent fetch threads                — default: 32
    --output-dir      Where to save model + logs              — default: ./model_output
    --failed-log      Path to failed URLs log                 — default: ./failed_urls.log
    --seed            Random seed                             — default: 42
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

# ── TensorFlow ────────────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress info/warning noise
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# ─────────────────────────────────────────────────────────────────────────────
# HTML fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_html(domain: str, timeout: int) -> str | None:
    """Fetch raw HTML for a domain using curl. Tries https then http."""
    for scheme in ("https", "http"):
        url = f"{scheme}://{domain}"
        try:
            result = subprocess.run(
                [
                    "curl",
                    "--silent",
                    "--max-time", str(timeout),
                    "--location",           # follow redirects
                    "--max-redirs", "5",
                    "--user-agent", "Mozilla/5.0 (compatible; research-bot/1.0)",
                    "--compressed",
                    url,
                ],
                capture_output=True,
                timeout=timeout + 5,        # subprocess hard kill after curl timeout
            )
            if result.returncode == 0 and result.stdout:
                html = result.stdout.decode("utf-8", errors="replace").strip()
                if html:
                    return html
        except (subprocess.TimeoutExpired, Exception):
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path, total: int, seed: int) -> list[tuple[str, int]]:
    """
    Load domain/label pairs from CSV, shuffle, and cap at `total`.
    Returns list of (domain, label) where label 1=malicious, 0=benign.
    """
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row["domain"].strip()
            label  = 1 if row["label"].strip().lower() == "malicious" else 0
            if domain:
                rows.append((domain, label))

    random.seed(seed)
    random.shuffle(rows)
    return rows[:total]


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────

def build_model(max_tokens: int, max_len: int) -> tuple:
    """
    Simple character/word-level model:
      TextVectorization → Embedding → GlobalAveragePooling → Dense
    Returns (vectorizer, model).
    """
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_len,
    )

    inputs  = tf.keras.Input(shape=(1,), dtype=tf.string)
    x       = vectorizer(inputs)
    x       = layers.Embedding(input_dim=max_tokens, output_dim=64, mask_zero=True)(x)
    x       = layers.GlobalAveragePooling1D()(x)
    x       = layers.Dense(64, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return vectorizer, model


# ─────────────────────────────────────────────────────────────────────────────
# Batch fetching (multithreaded)
# ─────────────────────────────────────────────────────────────────────────────

# Thread-safe counter for progress reporting
_print_lock = threading.Lock()


def _fetch_one(
    idx: int,
    total: int,
    domain: str,
    label: int,
    timeout: int,
    failed_log: Path,
) -> tuple[str, int, str | None]:
    """Fetch a single domain and return (domain, label, html|None)."""
    html = fetch_html(domain, timeout)
    status = f"OK ({len(html):,} chars)" if html else "FAILED"
    with _print_lock:
        print(f"  [{idx:>{len(str(total))}}/{total}] {domain} ... {status}")
    if not html:
        with _print_lock:
            with failed_log.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()}\t{domain}\n")
    return domain, label, html


def fetch_batch(
    batch: list[tuple[str, int]],
    timeout: int,
    failed_log: Path,
    workers: int = 32,
) -> tuple[list[str], list[int]]:
    """
    Fetch HTML for every domain in the batch concurrently.
    Results are re-ordered to match the original batch order.
    Returns (html_texts, labels).
    """
    total = len(batch)
    results: dict[str, tuple[int, str | None]] = {}  # domain → (label, html)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_one, i, total, domain, label, timeout, failed_log): domain
            for i, (domain, label) in enumerate(batch, 1)
        }
        for future in as_completed(futures):
            domain, label, html = future.result()
            results[domain] = (label, html)

    # Rebuild in original order so labels stay aligned
    texts, labels = [], []
    for domain, label in batch:
        _, html = results[domain]
        if html:
            texts.append(html)
            labels.append(label)
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train malicious-site classifier.")
    parser.add_argument("--dataset",    default="websites_labeled.csv")
    parser.add_argument("--total",      type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=20_000)
    parser.add_argument("--max-len",    type=int, default=1_000)
    parser.add_argument("--timeout",    type=int, default=10)
    parser.add_argument("--workers",    type=int, default=32,
                        help="Concurrent fetch threads per batch")
    parser.add_argument("--output-dir", default="./model_output")
    parser.add_argument("--failed-log", default="./failed_urls.log")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir   = Path(args.output_dir)
    failed_log   = Path(args.failed_log)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}")

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load & split into batches ─────────────────────────────────────────────
    print(f"Loading dataset from {dataset_path} ...")
    all_rows  = load_dataset(dataset_path, args.total, args.seed)
    n_batches = (len(all_rows) + args.batch_size - 1) // args.batch_size
    batches   = [
        all_rows[i * args.batch_size : (i + 1) * args.batch_size]
        for i in range(n_batches)
    ]
    print(f"  {len(all_rows):,} domains → {n_batches} batch(es) of up to {args.batch_size:,}\n")

    # ── Fetch first batch to fit the vectorizer ───────────────────────────────
    print(f"=== Batch 1 / {n_batches} — fetching HTML (vectorizer warm-up) ===")
    texts_0, labels_0 = fetch_batch(batches[0], args.timeout, failed_log, args.workers)

    if not texts_0:
        sys.exit("First batch returned no HTML. Check connectivity and try again.")

    print("\nFitting text vectorizer on first batch ...")
    vectorizer, model = build_model(args.max_tokens, args.max_len)
    vectorizer.adapt(tf.constant(texts_0))
    model.summary()

    history_log = output_dir / "training_history.csv"
    with history_log.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["batch", "epoch", "loss", "accuracy", "auc",
                                 "val_loss", "val_accuracy", "val_auc"])

    # ── Training loop ─────────────────────────────────────────────────────────
    for batch_idx, batch in enumerate(batches):
        b_num = batch_idx + 1
        if batch_idx == 0:
            texts, labels = texts_0, labels_0   # already fetched
        else:
            print(f"\n=== Batch {b_num} / {n_batches} — fetching HTML ===")
            texts, labels = fetch_batch(batch, args.timeout, failed_log, args.workers)

        if not texts:
            print(f"  Batch {b_num} produced no usable HTML — skipping.")
            continue

        # Build tf.data dataset
        X = tf.constant(texts)
        y = tf.constant(labels, dtype=tf.float32)
        dataset = (
            tf.data.Dataset.from_tensor_slices((X, y))
            .shuffle(len(texts), seed=args.seed)
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_size  = max(1, int(len(texts) * 0.15))
        val_ds    = dataset.take(val_size // 32 + 1)
        train_ds  = dataset.skip(val_size // 32 + 1)

        ckpt_path = output_dir / f"ckpt_batch_{b_num:02d}.weights.h5"
        callbacks = [
            ModelCheckpoint(
                filepath=str(ckpt_path),
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
                verbose=0,
            ),
            EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ]

        print(f"\n--- Training on batch {b_num} ({len(texts):,} samples) ---")
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Append per-epoch metrics to log
        with history_log.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for ep, (lo, ac, au, vlo, vac, vau) in enumerate(zip(
                hist.history.get("loss", []),
                hist.history.get("accuracy", []),
                hist.history.get("auc", []),
                hist.history.get("val_loss", []),
                hist.history.get("val_accuracy", []),
                hist.history.get("val_auc", []),
            ), 1):
                w.writerow([b_num, ep,
                             round(lo, 4), round(ac, 4), round(au, 4),
                             round(vlo, 4), round(vac, 4), round(vau, 4)])

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = output_dir / "model_final"
    print(f"\nSaving final model to {final_path} ...")

    # Wrap vectorizer + model into a single end-to-end SavedModel
    inputs  = tf.keras.Input(shape=(1,), dtype=tf.string, name="raw_html")
    outputs = model(vectorizer(inputs))
    end2end = tf.keras.Model(inputs, outputs, name="malicious_site_classifier")
    end2end.save(str(final_path))

    print(f"\n✓ Training complete.")
    print(f"  Final model : {final_path}")
    print(f"  History log : {history_log}")
    print(f"  Failed URLs : {failed_log}")
    print(f"\nTo score a new site:")
    print(f"  loaded = tf.keras.models.load_model('{final_path}')")
    print(f"  score  = loaded.predict([['<html>...your html...</html>']])[0][0]")
    print(f"  # score close to 1.0 = malicious, close to 0.0 = benign")


if __name__ == "__main__":
    main()