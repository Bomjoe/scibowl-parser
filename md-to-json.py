#!/usr/bin/env python3
from pathlib import Path
import re, json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Instantiate vLLM
llm = LLM(model=MODEL_ID,
          dtype="bfloat16",              # or "float16" depending on GPU
          tensor_parallel_size=1,
          trust_remote_code=True,
          mamba_ssm_cache_dtype="float32")        # >1 if you have multi‑GPU

# Sampling / generation parameters
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=8192,
    stop=None,
    skip_special_tokens=True,
)

# ------------------------------------------------------------
# Read one markdown file and build chat messages
# ------------------------------------------------------------
sys_path = Path("nemotron_sysprompt")
md_path  = Path("markdown/scibowl/Cast/1.mmd")

md_root = Path("markdown/")
md_files = [p for p in md_root.rglob("*.mmd") if p.is_file()]


system_prompt = sys_path.read_text(encoding="utf-8")

batch_inputs = []
for file in md_files:
    user_content  = file.read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_inputs.append(prompt)
    
    
# Generate output (synchronous single batch;
# later you can make a list of dicts for many files)
outputs = llm.generate(batch_inputs, sampling_params)
print(outputs)
# vLLM returns a list of RequestOutput objects
raw_text = outputs[0].outputs[0].text
# ------------------------------------------------------------
# Clean up model reasoning tags and ensure valid JSON
# ------------------------------------------------------------
def strip_think_blocks(s: str) -> str:
    return re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", s,
                  flags=re.DOTALL | re.IGNORECASE).strip()


try:
    final_text = json.dumps(
        json.loads(strip_think_blocks(raw_text)),
        ensure_ascii=False, indent=2)
except Exception:
    # if model produced trailing text, try to extract a JSON object
    m = re.search(r"(\{.*\}|\[.*\])", raw_text, re.DOTALL)
    candidate = m.group(1) if m else raw_text
    final_text = json.dumps(json.loads(strip_think_blocks(candidate)),
                            ensure_ascii=False, indent=2)

# Write to file
out_path = Path("json/scibowl/Cast/1.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(final_text, encoding="utf-8")
print(f"Saved output to {out_path}")