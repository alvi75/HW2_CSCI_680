from transformers import BertForMaskedLM

model_path = './output_model_hf'

model = BertForMaskedLM.from_pretrained(model_path)

model.save_pretrained(model_path)

print("Model saved in PyTorch format at:", model_path)
