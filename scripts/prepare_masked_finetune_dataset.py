import re
import json

input_file = "./data/code_search_net/code_finetune_dataset.jsonl"
output_file = "./data/code_search_net/code_finetune_masked_dataset.jsonl"

def mask_if_statements(code):
    return re.sub(r'\bif\b\s*\(.*?\)\s*:', '[MASK_IF]', code)

def process_dataset(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            masked_code = mask_if_statements(data['code'])
            data['masked_code'] = masked_code
            outfile.write(json.dumps(data) + "\n")
        print(f"Masked dataset saved to {output_file}")

process_dataset(input_file, output_file)
