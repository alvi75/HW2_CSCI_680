import os
import torch
from transformers import BertTokenizerFast, BertForMaskedLM, Trainer, TrainingArguments
from datasets import load_dataset

MODEL_PATH = './output_model_hf'
TOKENIZER_PATH = './tokenizer'
TRAIN_DATA_PATH = './data/code_search_net/code_finetune_masked_dataset.jsonl'
MAX_LENGTH = 512

tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)

train_dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")

def tokenize_function(examples):
    inputs = tokenizer(examples['masked_code'], padding='max_length', truncation=True, max_length=MAX_LENGTH)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir=MODEL_PATH,           
    overwrite_output_dir=True,       
    num_train_epochs=3,              
    per_device_train_batch_size=8,   
    save_steps=500,                  
    save_total_limit=2,              
    logging_dir='./logs',            
    logging_steps=50,                
    prediction_loss_only=True,       
    evaluation_strategy="no"         
)

trainer = Trainer(
    model=model,                         
    args=training_args,                 
    train_dataset=tokenized_dataset      
)

trainer.train()

trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(TOKENIZER_PATH)

print(f"Training complete. Model saved to {MODEL_PATH}.")
