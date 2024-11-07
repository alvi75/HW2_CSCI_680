from datasets import load_dataset
import json

output_file = "code_pretrain_dataset.jsonl"
target_count = 150000 

# Load the dataset
dataset = load_dataset("code_search_net", "python")

def save_to_jsonl(data, file_path):
    with open(file_path, "a") as file:
        for sample in data:
            file.write(json.dumps(sample) + "\n")

def create_code_dataset(dataset, output_file, target_count):
    code_samples = []
    count = 0
    
    for sample in dataset['train']:
        if "whole_func_string" in sample and sample["whole_func_string"].strip():
            code_entry = {
                "code": sample["whole_func_string"].strip(),
                "docstring": sample.get("func_documentation_string", "").strip()
            }
            code_samples.append(code_entry)
            count += 1

            if count % 1000 == 0:
                save_to_jsonl(code_samples, output_file)
                code_samples = []
                print(f"Saved {count} samples...")

            if count >= target_count:
                break

    if code_samples:
        save_to_jsonl(code_samples, output_file)
    print(f"Dataset creation complete. Total samples saved: {count}")

create_code_dataset(dataset, output_file, target_count)
