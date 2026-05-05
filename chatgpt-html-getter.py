import csv
import os
import requests
from concurrent.futures import ThreadPoolExecutor

INPUT_FILE = "websites_labeled.csv"
OUTPUT_DIR = "output"
TIMEOUT = 5
MAX_WORKERS = 20

def fetch_site(domain, label):
    url = f"http://{domain}"

    try:
        response = requests.get(
            url,
            timeout=TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if response.status_code != 200:
            return

        # create folder
        folder = os.path.join(OUTPUT_DIR, label)
        os.makedirs(folder, exist_ok=True)

        # safe filename
        filename = domain.replace("/", "_")
        filepath = os.path.join(folder, f"{filename}.txt")

        with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
            f.write(response.text)

        print(f"[✓] Saved: {domain}")

    except Exception:
        # skip silently (or print if debugging)
        pass


def main():
    tasks = []

    with open(INPUT_FILE, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            domain = row["domain"].strip()
            label = row["label"].strip()
            tasks.append((domain, label))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for domain, label in tasks:
            executor.submit(fetch_site, domain, label)


if __name__ == "__main__":
    main()