#!/usr/bin/env python3
import asyncio
import aiohttp
from pathlib import Path
import re, json

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
MODEL = "qwen/qwen3-235b-a22b-2507:floor"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = Path("openrouterkey.env").read_text().strip()

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

CONCURRENCY = 1000          # how many concurrent requests to run
MAX_RETRIES = 10            # total retry attempts on 429/5xx

# -----------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------
def strip_think_blocks(s: str) -> str:
    return re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>",
                  "", s, flags=re.DOTALL | re.IGNORECASE).strip()

def fix_single_backslashes(text: str) -> str:
    return re.sub(r'\\[^\\nt]', "\\\\", text)

def removeJsonMarkers(s: str) -> str:
    return re.sub(r'```json\n|```', "", s)



# -----------------------------------------------------------------
# ASYNC REQUEST FUNCTION
# -----------------------------------------------------------------
async def call_openrouter_chat(session: aiohttp.ClientSession, messages, fname: str):
    """Make one async OpenRouter call with retry & backoff."""
    payload = {
        "model": MODEL, 
        "messages": messages,
        "provider": {
            "require_parameters": true
        },
        "max_tokens": 16384
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(API_URL, headers=HEADERS, json=payload, timeout=600) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]

                if resp.status in {429, 500, 502, 503, 504}:
                    wait = 5 * (attempt + 1)
                    print(f"{fname}: rate/busy ({resp.status}); sleeping {wait}s …")
                    await asyncio.sleep(wait)
                    continue

                # Unexpected error → raise immediately
                txt = await resp.text()
                raise RuntimeError(f"{fname}: HTTP {resp.status} – {txt[:300]}")

        except asyncio.TimeoutError:
            wait = 5 * (attempt + 1)
            print(f"{fname}: timeout; retrying in {wait}s …")
            await asyncio.sleep(wait)
            continue

    raise RuntimeError(f"{fname}: failed after {MAX_RETRIES} retries")

# -----------------------------------------------------------------
# MAIN TASK PER FILE
# -----------------------------------------------------------------
async def process_file(session: aiohttp.ClientSession, sys_prompt: str, file_path: Path):
    user_content = file_path.read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_content},
    ]

    print(f"Submitting {file_path}")
    try:
        raw_text = await call_openrouter_chat(session, messages, str(file_path))
    except Exception as e:
        print(f"{file_path} failed: {e}")
        raw_text = f"Failure: {e}"

    cleaned = strip_think_blocks(raw_text)
    cleaned = fix_single_backslashes(cleaned)
    cleaned = removeJsonMarkers(cleaned)

    try:
        final_text = json.dumps(json.loads(cleaned),
                                ensure_ascii=False, indent=2)
    except Exception:
        print(f"{file_path} parsing failed; storing raw text")
        final_text = "Failure:\n" + cleaned

    out_path = Path("json") / file_path.relative_to("markdown")
    out_path = out_path.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(final_text, encoding="utf-8")
    print(f"Saved {out_path}")

# -----------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------
async def main():
    sys_prompt = Path("nemotron_sysprompt").read_text(encoding="utf-8")

    md_root = Path("markdown/scibowl")
    md_files = [p for p in md_root.rglob("*.mmd") if p.is_file()]

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def sem_task(file_path):
            async with sem:
                await process_file(session, sys_prompt, file_path)

        await asyncio.gather(*(sem_task(f) for f in md_files))

if __name__ == "__main__":
    asyncio.run(main())