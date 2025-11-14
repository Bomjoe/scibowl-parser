from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm
from pathlib import Path
from pdf2image import convert_from_path
from contextlib import redirect_stdout

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)


prompt = "<image>\n<|grounding|>Convert the document to latex. "



pdf_root = Path("pdfs")

files = [p for p in pdf_root.rglob("*") if p.is_file()]
files = [Path("pdfs/scibowl/Cast/1.pdf")]
with tqdm(total=len(files), desc="Performing OCR", ncols=80) as pbar:
    for pdf_path in files:
        image_dir = Path(f"images/{str(pdf_path)[4:-4]}/")
        image_dir.mkdir(parents=True, exist_ok=True)

        images = convert_from_path(pdf_path, dpi=300)

        pdftext = ""
        for i, img in enumerate(images[0:1]):
            jpg_path = f"{image_dir}/page_{i+1}.jpg"
            img.save(jpg_path, "JPEG")
            print(f"Saved {jpg_path}")
            

            with open(os.devnull, "w") as f, redirect_stdout(f):
                res = model.infer(tokenizer, prompt=prompt, 
                        image_file=jpg_path, output_path = jpg_path+".out", 
                        base_size = 1024, image_size = 640, 
                        crop_mode=False, save_results = True, test_compress = True)
            
            with open(jpg_path+".out" + "/result.mmd", "r", encoding="utf-8") as ocr:
                pdftext += ocr.read()

        ouput_path = Path(f"markdown/{str(pdf_path)[4:-4]}.mmd")
        ouput_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ouput_path, "w", encoding="utf-8") as output_file:
            output_file.write(pdftext)
        pbar.update(1)