import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DebertaV2Tokenizer
from safetensors.torch import load_file
from models import DebertaPureRegressor, LlamaEmotionalProjector
from config import *

# # Initialize models
# de_tokenizer = DebertaV2Tokenizer.from_pretrained(DEBERTA_PATH)
# de_model = DebertaPureRegressor(DEBERTA_PATH).to(DEVICE)
# sf_path = os.path.join(DEBERTA_PATH, "model.safetensors")
# if os.path.exists(sf_path):
#     de_model.load_state_dict(load_file(sf_path, device=DEVICE))
# de_model.eval().requires_grad_(False)

# llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
# llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")
# llama_model.eval().requires_grad_(False)

# projector = LlamaEmotionalProjector(llama_dim=3072).to(DEVICE).to(torch.bfloat16)
# projector.load_state_dict(torch.load(PROJECTOR_WEIGHTS, map_location=DEVICE))
# projector.eval().requires_grad_(False)

print("🧠 Loading NEW Pure DeBERTa Encoder...")
de_model = DebertaPureRegressor(DEBERTA_PATH).to(DEVICE)
sf_path = os.path.join(DEBERTA_PATH, "model.safetensors")
if os.path.exists(sf_path):
    de_model.load_state_dict(load_file(sf_path, device=DEVICE))
    print("✅ Loaded DeBERTa weights.")
de_model = de_model.to(torch.float32).eval()
de_tokenizer = DebertaV2Tokenizer.from_pretrained(DEBERTA_PATH)

print("📦 Loading Llama 3.2 3B...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
)
llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_PATH, quantization_config=bnb_config, 
    torch_dtype=torch.bfloat16, device_map="auto"
)
print("✅✅ Loaded Llama weights.")
llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)

print("🔗 Loading Projector...")
projector = LlamaEmotionalProjector(llama_dim=3072).to(DEVICE).to(torch.bfloat16)
projector.load_state_dict(torch.load(PROJECTOR_WEIGHTS, map_location=DEVICE))
print("✅✅✅ Complete load model")
projector.eval()