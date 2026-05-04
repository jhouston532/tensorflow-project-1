"""
evaluate_model.py

Evaluates the trained malicious-site classifier against the remainder of the
shuffled dataset that was NOT used during training (i.e. everything after the
first 20,000 entries). Fetches HTML via curl, runs inference, and produces a
detailed accuracy report.

Usage:
    python evaluate_model.py --dataset websites_labeled.csv

Optional flags:
    --dataset       Path to normalized CSV (domain, label)  — default: websites_labeled.csv
    --model-dir     Path to saved model                     — default: ./model_output/model_final
    --skip          How many rows were used for training     — default: 20000
    --timeout       Curl timeout per site in seconds         — default: 10
    --threshold     Score threshold for malicious (0-1)      — default: 0.5
    --failed-log    Path to failed URLs log                  — default: ./failed_eval_urls.log
    --output        Path for per-site results CSV            — default: ./eval_results.csv
    --seed          Must match the seed used in training     — default: 42
    --batch-size    Fetch + score in chunks (memory)         — default: 500
"""

import argparse
import csv
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared logic with train_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_full_dataset(path: Path, seed: int) -> list[tuple[str, int]]:
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
    return rows


def fetch_html(domain: str, timeout: int) -> str | None:
    for scheme in ("https", "http"):
        url = f"{scheme}://{domain}"
        try:
            result = subprocess.run(
                [
                    "curl", "--silent",
                    "--max-time", str(timeout),
                    "--location", "--max-redirs", "5",
                    "--user-agent", "Mozilla/5.0 (compatible; research-bot/1.0)",
                    "--compressed", url,
                ],
                capture_output=True,
                timeout=timeout + 5,
            )
            if result.returncode == 0 and result.stdout:
                html = result.stdout.decode("utf-8", errors="replace").strip()
                if html:
                    return html
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    true_labels: list[int],
    pred_labels: list[int],
    scores: list[float],
    threshold: float,
) -> dict:
    tp = sum(t == 1 and p == 1 for t, p in zip(true_labels, pred_labels))
    tn = sum(t == 0 and p == 0 for t, p in zip(true_labels, pred_labels))
    fp = sum(t == 0 and p == 1 for t, p in zip(true_labels, pred_labels))
    fn = sum(t == 1 and p == 0 for t, p in zip(true_labels, pred_labels))
    n  = len(true_labels)

    accuracy  = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)
    fpr       = fp / (fp + tn) if (fp + tn) else 0   # false positive rate

    # AUC via trapezoidal rule (no sklearn needed)
    paired   = sorted(zip(scores, true_labels), key=lambda x: -x[0])
    auc, tp_c, fp_c = 0.0, 0, 0
    prev_fpr, prev_tpr = 0.0, 0.0
    pos = sum(true_labels)
    neg = n - pos
    for sc, lb in paired:
        if lb == 1:
            tp_c += 1
        else:
            fp_c += 1
        tpr = tp_c / pos if pos else 0
        cur_fpr = fp_c / neg if neg else 0
        auc += (cur_fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr, prev_tpr = cur_fpr, tpr

    return dict(
        n=n, tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=accuracy, precision=precision,
        recall=recall, f1=f1, fpr=fpr, auc=auc,
        threshold=threshold,
    )


def print_report(m: dict, n_malicious: int, n_benign: int) -> None:
    width = 52
    bar   = "─" * width
    print(f"\n{'═' * width}")
    print(f"  EVALUATION REPORT")
    print(f"{'═' * width}")
    print(f"  Sites evaluated   : {m['n']:>8,}")
    print(f"  Malicious (truth) : {n_malicious:>8,}")
    print(f"  Benign    (truth) : {n_benign:>8,}")
    print(f"  Score threshold   : {m['threshold']:>8.2f}")
    print(bar)
    print(f"  Accuracy          : {m['accuracy']:>8.2%}")
    print(f"  AUC-ROC           : {m['auc']:>8.4f}")
    print(f"  Precision         : {m['precision']:>8.2%}")
    print(f"  Recall            : {m['recall']:>8.2%}")
    print(f"  F1 Score          : {m['f1']:>8.4f}")
    print(f"  False Positive Rate: {m['fpr']:>7.2%}")
    print(bar)
    print(f"  True  Positives   : {m['tp']:>8,}  (malicious caught)")
    print(f"  True  Negatives   : {m['tn']:>8,}  (benign correctly cleared)")
    print(f"  False Positives   : {m['fp']:>8,}  (benign flagged as malicious)")
    print(f"  False Negatives   : {m['fn']:>8,}  (malicious missed)")
    print(f"{'═' * width}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate malicious-site classifier.")
    parser.add_argument("--dataset",    default="websites_labeled.csv")
    parser.add_argument("--model-dir",  default="./model_output/model_final")
    parser.add_argument("--skip",       type=int,   default=20_000)
    parser.add_argument("--timeout",    type=int,   default=10)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--failed-log", default="./failed_eval_urls.log")
    parser.add_argument("--output",     default="./eval_results.csv")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--batch-size", type=int,   default=500)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    model_path   = Path(args.model_dir)
    failed_log   = Path(args.failed_log)
    output_path  = Path(args.output)

    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}")
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}. Run train_model.py first.")

    # ── Load shuffled list and take the eval portion ──────────────────────────
    print(f"Loading dataset (seed={args.seed}) ...")
    all_rows  = load_full_dataset(dataset_path, args.seed)
    eval_rows = all_rows[args.skip:]

    if not eval_rows:
        sys.exit(
            f"No rows left for evaluation after skipping {args.skip:,}. "
            f"Dataset only has {len(all_rows):,} entries."
        )

    print(f"  Total rows     : {len(all_rows):,}")
    print(f"  Training rows  : {args.skip:,}  (skipped)")
    print(f"  Evaluation rows: {len(eval_rows):,}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))
    print("  Model loaded.\n")

    # ── Prepare results CSV ───────────────────────────────────────────────────
    with output_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["domain", "true_label", "predicted_label", "score", "correct"]
        )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_true, all_pred, all_scores = [], [], []
    fetched_total = 0
    failed_total  = 0

    batches = [
        eval_rows[i : i + args.batch_size]
        for i in range(0, len(eval_rows), args.batch_size)
    ]

    for b_idx, batch in enumerate(batches, 1):
        print(f"=== Eval batch {b_idx} / {len(batches)} "
              f"({len(batch)} sites) ===")

        html_list, label_list, domain_list = [], [], []

        for i, (domain, label) in enumerate(batch, 1):
            print(f"  [{i:>4}/{len(batch)}] {domain} ... ", end="", flush=True)
            html = fetch_html(domain, args.timeout)
            if html:
                html_list.append(html)
                label_list.append(label)
                domain_list.append(domain)
                fetched_total += 1
                print(f"OK ({len(html):,} chars)")
            else:
                failed_total += 1
                print("FAILED")
                with failed_log.open("a", encoding="utf-8") as f:
                    f.write(f"{datetime.now().isoformat()}\t{domain}\n")

        if not html_list:
            print("  No HTML fetched in this batch — skipping inference.\n")
            continue

        # ── Inference ─────────────────────────────────────────────────────────
        print(f"\n  Running inference on {len(html_list)} sites ...")
        X      = tf.constant(html_list)
        scores = model.predict(X, batch_size=32, verbose=0).flatten().tolist()
        preds  = [1 if s >= args.threshold else 0 for s in scores]

        all_true.extend(label_list)
        all_pred.extend(preds)
        all_scores.extend(scores)

        # Append to per-site results CSV
        with output_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for domain, true_lb, pred_lb, score in zip(
                domain_list, label_list, preds, scores
            ):
                true_str = "malicious" if true_lb == 1 else "benign"
                pred_str = "malicious" if pred_lb == 1 else "benign"
                correct  = true_lb == pred_lb
                w.writerow([domain, true_str, pred_str, f"{score:.4f}", correct])

        # Running accuracy after each batch
        correct_so_far = sum(t == p for t, p in zip(all_true, all_pred))
        print(f"  Running accuracy: {correct_so_far / len(all_true):.2%} "
              f"({correct_so_far:,} / {len(all_true):,})\n")

    # ── Final report ──────────────────────────────────────────────────────────
    if not all_true:
        print("No sites were successfully evaluated.")
        return

    metrics    = compute_metrics(all_true, all_pred, all_scores, args.threshold)
    n_malicious = sum(all_true)
    n_benign    = len(all_true) - n_malicious

    print_report(metrics, n_malicious, n_benign)

    print(f"  Per-site results : {output_path}")
    print(f"  Failed URLs log  : {failed_log}")
    print(f"  Fetch failures   : {failed_total:,} "
          f"(not counted in metrics)\n")


if __name__ == "__main__":
    main()