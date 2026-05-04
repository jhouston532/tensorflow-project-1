"""
normalize_websites.py

Merges a malicious website list (.txt, one domain per line) and a benign
website list (.csv with a 'Domain' column) into a single normalized CSV.

Usage:
    python normalize_websites.py \
        --malicious malicious.txt \
        --benign benign.csv \
        --output websites_labeled.csv

Optional flags:
    --malicious-col   Column name in a CSV-format malicious file (default: plain txt)
    --benign-col      Column name for domains in the benign CSV (default: "Domain")
    --shuffle         Randomly shuffle the output rows
    --seed            Random seed for reproducibility (used with --shuffle)
"""

import argparse
import csv
import random
import sys
from pathlib import Path


def load_malicious_txt(path: Path) -> list[str]:
    """Load domains from a plain-text file (one domain per line)."""
    domains = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            domain = line.strip()
            if domain and not domain.startswith("#"):
                domains.append(domain.lower())
    return domains


def load_benign_csv(path: Path, domain_col: str) -> list[str]:
    """Load domains from a CSV file using the specified column name."""
    domains = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if domain_col not in (reader.fieldnames or []):
            available = ", ".join(reader.fieldnames or [])
            sys.exit(
                f"Column '{domain_col}' not found in {path.name}. "
                f"Available columns: {available}"
            )
        for row in reader:
            domain = row[domain_col].strip()
            if domain:
                domains.append(domain.lower())
    return domains


def write_output(rows: list[tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "label"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Normalize website label datasets.")
    parser.add_argument("--malicious", required=True, help="Path to malicious domains .txt file")
    parser.add_argument("--benign", required=True, help="Path to benign domains .csv file")
    parser.add_argument("--output", default="websites_labeled.csv", help="Output CSV path")
    parser.add_argument("--benign-col", default="Domain", help="Column name for domains in benign CSV")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used with --shuffle)")
    args = parser.parse_args()

    malicious_path = Path(args.malicious)
    benign_path = Path(args.benign)
    output_path = Path(args.output)

    if not malicious_path.exists():
        sys.exit(f"Malicious file not found: {malicious_path}")
    if not benign_path.exists():
        sys.exit(f"Benign file not found: {benign_path}")

    malicious_domains = load_malicious_txt(malicious_path)
    benign_domains = load_benign_csv(benign_path, args.benign_col)

    # Deduplicate within each list, preserving order
    malicious_domains = list(dict.fromkeys(malicious_domains))
    benign_domains = list(dict.fromkeys(benign_domains))

    # Flag any overlap so the user is aware
    overlap = set(malicious_domains) & set(benign_domains)
    if overlap:
        print(f"Warning: {len(overlap)} domain(s) appear in both lists. "
              f"They will be labelled 'malicious'. Examples: {list(overlap)[:5]}")
        benign_domains = [d for d in benign_domains if d not in overlap]

    rows = [(d, "malicious") for d in malicious_domains] + \
           [(d, "benign")    for d in benign_domains]

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(rows)

    write_output(rows, output_path)

    print(f"Done.")
    print(f"  Malicious : {len(malicious_domains):,}")
    print(f"  Benign    : {len(benign_domains):,}")
    print(f"  Total     : {len(rows):,}")
    print(f"  Output    : {output_path}")


if __name__ == "__main__":
    main()