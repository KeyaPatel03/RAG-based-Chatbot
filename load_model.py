import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
 
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# set pad token on tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)

pad_id = tokenizer.pad_token_id

# set on model config
model.config.pad_token_id = pad_id

# set on generation_config too (this is what generate() uses)
if hasattr(model, "generation_config") and model.generation_config is not None:
    model.generation_config.pad_token_id = pad_id

print("Model loaded successfully!")
