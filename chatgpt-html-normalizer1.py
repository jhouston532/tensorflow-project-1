#!/usr/bin/env python3

"""
USE: 
    python normalize_html.py --input-dir output
"""



import argparse
import os
from pathlib import Path
from bs4 import BeautifulSoup, Comment


# ─────────────────────────────────────────────────────────────────────────────
# Normalization logic
# ─────────────────────────────────────────────────────────────────────────────

def normalize_html(html: str, lowercase: bool = True) -> str:
    """
    Normalize HTML into clean, model-friendly text.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Extract visible text
    text = soup.get_text(separator=" ")

    # Normalize whitespace
    text = " ".join(text.split())

    if lowercase:
        text = text.lower()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# File processing
# ─────────────────────────────────────────────────────────────────────────────

def process_file(path: Path, overwrite: bool = False):
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return

    normalized = normalize_html(html)

    output_path = path.with_name(path.stem + "_normalized.html")

    if output_path.exists() and not overwrite:
        print(f"[SKIP] {output_path} already exists")
        return

    try:
        output_path.write_text(normalized, encoding="utf-8")
        print(f"[OK] {path} → {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write {output_path}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Directory traversal
# ─────────────────────────────────────────────────────────────────────────────

def process_directory(root: Path, overwrite: bool = False):
    if not root.exists():
        print(f"[ERROR] Directory does not exist: {root}")
        return

    html_files = list(root.rglob("*.html"))

    print(f"Found {len(html_files)} HTML files in {root}")

    for path in html_files:
        # Skip already normalized files
        if path.name.endswith("_normalized.html"):
            continue

        process_file(path, overwrite=overwrite)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Normalize HTML files for ML training")
    parser.add_argument(
        "--input-dir",
        default="output",
        help="Root directory containing malicious/benign folders"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing normalized files"
    )

    args = parser.parse_args()

    root = Path(args.input_dir)

    print(f"Processing directory: {root}")
    process_directory(root, overwrite=args.overwrite)

    print("\n✓ Normalization complete.")


if __name__ == "__main__":
    main()