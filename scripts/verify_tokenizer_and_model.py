import torch
from transformers import BertTokenizerFast, BertForMaskedLM

tokenizer_path = './tokenizer'
model_path = './output_model_hf'

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
print("Tokenizer loaded successfully.")
print(f"Pad token ID: {tokenizer.pad_token_id}")
print(f"CLS token ID: {tokenizer.cls_token_id}")
print(f"Mask token ID: {tokenizer.mask_token_id}")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForMaskedLM.from_pretrained(model_path)
model.to(device)
print("Model loaded successfully and moved to device:", device)
