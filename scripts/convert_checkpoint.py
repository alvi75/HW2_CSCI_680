import torch
from transformers import BertForMaskedLM

checkpoint_path = "output_model/epoch=2-step=56250.ckpt"
hf_model_save_path = "output_model_hf"

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

model.load_state_dict(checkpoint['state_dict'], strict=False)

model.save_pretrained(hf_model_save_path)
print(f"Model saved in Hugging Face format at {hf_model_save_path}")
