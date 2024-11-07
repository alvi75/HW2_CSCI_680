import json

vocab_json_path = 'tokenizer/vocab.json'

vocab_txt_path = 'tokenizer/vocab.txt'

with open(vocab_json_path, 'r') as json_file:
    vocab = json.load(json_file)

with open(vocab_txt_path, 'w') as txt_file:
    for token in vocab.keys():
        txt_file.write(token + '\n')

print(f"Conversion complete! 'vocab.txt' saved at {vocab_txt_path}")
