import csv
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_CSV = "websites_labeled.csv"
OUTPUT_DIR = "output"
TIMEOUT = 5
MAX_WORKERS = 20  # adjust based on your system/network

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive"
}


def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "benign"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "malicious"), exist_ok=True)


def normalize_url(domain):
    if not domain.startswith(("http://", "https://")):
        return "http://" + domain
    return domain


def safe_filename(domain):
    return domain.replace("/", "_").replace(":", "_")


def fetch_and_save(domain, label):
    url = normalize_url(domain)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT,
            allow_redirects=True
        )

        if 200 <= response.status_code < 400:
            filename = safe_filename(domain) + ".html"
            filepath = os.path.join(OUTPUT_DIR, label, filename)

            with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
                f.write(response.text)

            return f"[OK] {domain}"
        else:
            return f"[FAIL] {domain} -> {response.status_code}"

    except requests.RequestException as e:
        return f"[ERROR] {domain} -> {e}"


def main():
    ensure_dirs()

    tasks = []

    with open(INPUT_CSV, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            domain = row["domain"].strip()
            label = row["label"].strip().lower()

            if label not in ["benign", "malicious"]:
                print(f"[SKIP] {domain} -> invalid label")
                continue

            tasks.append((domain, label))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(fetch_and_save, domain, label)
            for domain, label in tasks
        ]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()