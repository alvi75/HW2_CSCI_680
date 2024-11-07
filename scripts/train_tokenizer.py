import argparse
from tokenizers import ByteLevelBPETokenizer

def train_tokenizer(input_path, output_path, vocab_size=50000):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[input_path], vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>"
    ])
    tokenizer.save_model(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tokenizer on code data.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the prepared data file")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the tokenizer")
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size for the tokenizer")
    args = parser.parse_args()
    train_tokenizer(args.input_path, args.output_path, args.vocab_size)
