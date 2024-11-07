from transformers import BertTokenizerFast

tokenizer_path = './tokenizer'

tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

special_tokens = {
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
    "unk_token": "[UNK]"
}
tokenizer.add_special_tokens(special_tokens)

tokenizer.save_pretrained(tokenizer_path)

print("Tokenizer files saved successfully in:", tokenizer_path)
