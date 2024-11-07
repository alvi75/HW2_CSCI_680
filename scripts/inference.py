import torch
from transformers import BertTokenizerFast, BertForMaskedLM
import pandas as pd

# Paths
MODEL_PATH = './output_model_hf'
TOKENIZER_PATH = './tokenizer'
OUTPUT_CSV_PATH_GENERATED = 'generated-testset.csv'
OUTPUT_CSV_PATH_PROVIDED = 'provided-testset.csv'

tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
model = BertForMaskedLM.from_pretrained(MODEL_PATH)
model.eval()

test_cases = [
    {"input": "def check_value(x):\n    [MASK] x > 10:\n        return 'High'\n    else:\n        return 'Low'", "expected_if": "if"},
    {"input": "def calculate_sum(a, b):\n    [MASK] a > b:\n        return a\n    else:\n        return b", "expected_if": "if"},
    {"input": "def is_positive(num):\n    [MASK] num >= 0:\n        return True\n    else:\n        return False", "expected_if": "if"},
    {"input": "for i in range(10):\n    [MASK] i % 2 == 0:\n        print(i)", "expected_if": "if"},
    {"input": "def grade_score(score):\n    [MASK] score > 90:\n        return 'A'\n    else:\n        return 'B'", "expected_if": "if"}
]

generated_data = []
provided_data = []

for case in test_cases:
    input_text = case["input"]
    expected_if = case["expected_if"]

    inputs = tokenizer(input_text, return_tensors='pt')
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    mask_logits = logits[0, mask_token_index, :].squeeze()
    
    top_k = torch.topk(mask_logits, 5)
    top_k_indices = top_k.indices
    top_k_scores = top_k.values

    predictions = [tokenizer.decode([idx]) for idx in top_k_indices]
    scores = [round(score.item(), 2) for score in top_k_scores]

    for pred, score in zip(predictions, scores):
        generated_data.append({
            "Input": input_text,
            "Expected if condition": expected_if,
            "Predicted if condition": pred,
            "Prediction score (0-100)": score
        })

    top_pred = predictions[0]
    top_score = scores[0]
    is_correct = (top_pred == expected_if)
    
    provided_data.append({
        "Input": input_text,
        "Correct (True/False)": is_correct,
        "Expected if condition": expected_if,
        "Predicted if condition": top_pred,
        "Prediction score (0-100)": top_score
    })

generated_df = pd.DataFrame(generated_data)
provided_df = pd.DataFrame(provided_data)

generated_df.to_csv(OUTPUT_CSV_PATH_GENERATED, index=False)
provided_df.to_csv(OUTPUT_CSV_PATH_PROVIDED, index=False)

print(f"Results saved to {OUTPUT_CSV_PATH_GENERATED} and {OUTPUT_CSV_PATH_PROVIDED}.")
