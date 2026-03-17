import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "beratcmn/Mistral-7B-v0.1-int8"  # or the exact v0 you’re using

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                # 8-bit weights via bitsandbytes (LLM.int8)
    llm_int8_threshold=6.0,           # default is usually fine
    llm_int8_enable_fp32_cpu_offload=False
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# torch_dtype controls the compute & KV cache dtype; with 8-bit weights this ensures fp16 KV.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,        # => fp16 KV cache + compute
    device_map="auto"                 # place layers on the Wave GPU automatically
)

# Optional: better attention kernels if available on your GPU/CUDA/PyTorch build
# model = model.to_bettertransformer()  # if you're using BT; or set attn_implementation="flash_attention_2" in from_pretrained if supported

prompt = "You are a helpful assistant."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
