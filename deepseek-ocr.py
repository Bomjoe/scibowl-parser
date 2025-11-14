#!/usr/bin/env python3
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from tqdm import tqdm
from pathlib import Path
from pdf2image import convert_from_path
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict
import os, locale
import io
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
MODEL_ID = 'deepseek-ai/DeepSeek-OCR'

llm = LLM(
    model=MODEL_ID,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor] 
)

prompt = "<image>\n<|grounding|>Convert the document to markdown."

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=16384,  # reduce if VRAM is tight
    skip_special_tokens=True,
    extra_args=dict(
        # Only used if you provided NGramPerReqLogitsProcessor above
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
    )
)

pdf_root = Path("pdfs/")
pdf_files = [p for p in pdf_root.rglob("*.pdf") if p.is_file()]

total_pages = 0
from PyPDF2 import PdfReader
for p in pdf_files:
    reader = PdfReader(str(p))
    n = len(reader.pages)
    total_pages += n

BATCH_SIZE = 600
DPI = 200


requests = []
page_meta = []  # (pdf_path, page_index) for result routing
pdf_outputs = defaultdict(list)

# ---- Run vLLM in large batches ----
with tqdm(total=total_pages, desc="Performing OCR", ncols=80) as pbar:
    for pdf_path in pdf_files:
        pages = convert_from_path(pdf_path, dpi=DPI)
        for i, img in enumerate(pages, start=1):
            requests.append({"prompt": prompt, "multi_modal_data": {"image": img}})
            page_meta.append((pdf_path, i))
            if len(requests) >= BATCH_SIZE:
                outputs = llm.generate(requests, sampling_params)
                for out, meta in zip(outputs, page_meta):
                    pdf_path, page_num = meta
                    text = out.outputs[0].text
                    pdf_outputs[pdf_path].append((page_num, text))
                pbar.update(len(requests))
                requests.clear()
                page_meta.clear()

    if len(requests)>0:
        outputs = llm.generate(requests, sampling_params)
        for out, meta in zip(outputs, page_meta):
            pdf_path, page_num = meta
            text = out.outputs[0].text
            pdf_outputs[pdf_path].append((page_num, text))
        pbar.update(len(requests))
        requests.clear()
        page_meta.clear()

# ---- Write one .mmd per PDF ----
for pdf_path, results in pdf_outputs.items():
    # Sort pages numerically just in case batches returned out of order
    results.sort(key=lambda x: x[0])
    combined = "".join(text for _, text in results)
    output_path = Path(f"markdown/{str(pdf_path)[4:-4]}.mmd")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)

print(f"Wrote {len(pdf_outputs)} markdown files.")
        