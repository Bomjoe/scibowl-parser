#!/usr/bin/env python3
import re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Path to your input file
input_path = Path("index.html")

# Read file contents
text = input_path.read_text(encoding="utf-8")

# Extract data-path values ending with .pdf
pdf_paths = re.findall(r'data-path="([^"]+\.pdf)"', text)

print(f"Found {len(pdf_paths)} PDF paths:")
for path in pdf_paths:
    print(path)

# The base part you want to preserve
base_prefix = "pdfs/scibowl"

# Directory where you'll download files
download_root = Path("pdfs")
base_url = "https://oly.mehvix.com/"


def download_pdf(pdf_path: str):
    """Download a single PDF and save it locally."""
    if not pdf_path.startswith(base_prefix):
        return f"Skipping non-matching path: {pdf_path}"

    try:
        relative_path = Path(pdf_path).relative_to("pdfs")
    except ValueError:
        return f"Invalid path (cannot make relative): {pdf_path}"

    output_path = download_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_url = f"{base_url}{pdf_path}"
    try:
        response = requests.get(file_url, timeout=15)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return f"✅ Saved: {output_path}"
    except Exception as e:
        return f"❌ Failed to download {file_url}: {e}"


# Thread pool for concurrent downloads
max_workers = 10  # Adjust as needed
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(download_pdf, pdf): pdf for pdf in pdf_paths}

    # Initialize tqdm progress bar
    with tqdm(total=len(pdf_paths), desc="Downloading PDFs", ncols=80) as pbar:
        for future in as_completed(futures):
            result = future.result()
            print(result)
            pbar.update(1)