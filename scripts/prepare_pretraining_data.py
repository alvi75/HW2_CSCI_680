import json
import argparse

def prepare_data(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            code = data.get("code")
            if code:
                prepared_data = {"text": code}
                outfile.write(json.dumps(prepared_data) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for pretraining.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file")
    args = parser.parse_args()
    prepare_data(args.input_path, args.output_path)
