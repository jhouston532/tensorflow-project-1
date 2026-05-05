import csv
import os
import asyncio
import aiohttp

INPUT_CSV = "websites_labeled.csv"
OUTPUT_DIR = "output"
TIMEOUT = 5
CONCURRENCY = 100  # tune this (50–500 depending on system)

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


async def fetch_and_save(session, semaphore, domain, label):
    url = normalize_url(domain)

    async with semaphore:
        try:
            async with session.get(url, timeout=TIMEOUT) as response:
                status = response.status

                if 200 <= status < 400:
                    text = await response.text(errors="ignore")

                    filename = safe_filename(domain) + ".html"
                    filepath = os.path.join(OUTPUT_DIR, label, filename)

                    with open(filepath, "w", encoding="utf-8", errors="ignore") as f:
                        f.write(text)

                    return f"[OK] {domain}"
                else:
                    return f"[FAIL] {domain} -> {status}"

        except Exception as e:
            return f"[ERROR] {domain} -> {e}"


async def main():
    ensure_dirs()

    tasks = []
    semaphore = asyncio.Semaphore(CONCURRENCY)

    timeout = aiohttp.ClientTimeout(total=TIMEOUT)

    connector = aiohttp.TCPConnector(
        limit=CONCURRENCY,
        ssl=False  # avoids SSL issues with sketchy domains
    )

    async with aiohttp.ClientSession(
        headers=HEADERS,
        timeout=timeout,
        connector=connector
    ) as session:

        with open(INPUT_CSV, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                domain = row["domain"].strip()
                label = row["label"].strip().lower()

                if label not in ["benign", "malicious"]:
                    print(f"[SKIP] {domain} -> invalid label")
                    continue

                task = asyncio.create_task(
                    fetch_and_save(session, semaphore, domain, label)
                )
                tasks.append(task)

        for future in asyncio.as_completed(tasks):
            result = await future
            print(result)


if __name__ == "__main__":
    asyncio.run(main())