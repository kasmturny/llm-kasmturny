import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./Qwen2-0.5B/")
model = AutoModelForCausalLM.from_pretrained("./WikiLLM/Weight/")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = transformers.pipeline("text-generation", model=model,
                             tokenizer=tokenizer, device=device,
                             max_length=50, truncation=True)
print("GENERATE:", pipe("中国", num_return_sequences=1)[0]["generated_text"])
