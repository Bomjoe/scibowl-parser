from pathlib import Path
from llama_cpp import Llama
import json 
import re

llm = Llama.from_pretrained(
	repo_id="Qwen/Qwen3-8B-GGUF",
	filename="Qwen3-8B-Q4_K_M.gguf",
    n_ctx=16192,
    n_gpu_layers=-1
)


print("bee")  # debug marker before load complete

# Build your conversation prompt exactly like before
messages = []
sysprompt = open("nemotron_sysprompt","r") 
markdown  = open("markdown/scibowl/Cast/1.mmd", "r") 

messages.append({"role": "system", "content": sysprompt.read()})
messages.append({"role": "user", "content": markdown.read()})

print("baa")  # debug marker

# Stream tokens as they are produced
output = []
for part in llm.create_chat_completion(
    messages,
    max_tokens=None,
    temperature=0.7,
    stream=True,
    # response_format={
    #     "type": "json_object",
    #     "schema": {
    #         "type": "object",
    #         "properties": {
    #             "team_name": {"type": "string"},
    #             "body": {"type": "string"}
    #             "choices": {
    #                 "W": {"type": "string"},
    #                 "X": {"type": "string"},
    #                 "Y": {"type": "string"},
    #                 "Z": {"type": "string"},
    #             }
    #         },
    #         "required": ["team_name"],
    #     },
    # }
):
    #print(part)
    delta = part["choices"][0]["delta"].get("content", "")
    print(delta, end="", flush=True)
    
    output.append(delta)

def strip_think_blocks(s: str) -> str:
    # Remove <think>...</think> (greedy across newlines)
    s = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", s, flags=re.DOTALL | re.IGNORECASE)
    return s

# Join the output text if you want to save it
final_text = json.dumps(json.loads(strip_think_blocks("".join(output))), ensure_ascii=False, indent=2)

output_path = Path("json/scibowl/Cast/1.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(final_text)
print(f"\n\nSaved output to {output_path}")