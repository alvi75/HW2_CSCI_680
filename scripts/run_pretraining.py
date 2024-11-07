import os
import torch
import transformers
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional

# Define constants
MAX_LENGTH = 512

class MLMDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 8):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        self.tokenizer = BertTokenizerFast(vocab_file="tokenizer/vocab.txt", merges_file="tokenizer/merges.txt")

        special_tokens_dict = {
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def setup(self, stage: Optional[str] = None):
        self.dataset = load_dataset("json", data_files=self.data_path, split="train")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
            )

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

class BERTMLM(pl.LightningModule):
    def __init__(self, tokenizer, learning_rate: float = 3e-5, weight_decay: float = 0.01, num_warmup_steps: int = 100, num_training_steps: int = 1000):
        super().__init__()
        self.tokenizer = tokenizer 
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

if __name__ == "__main__":
    data_path = "data/code_search_net/prepared_data.jsonl"
    output_dir = "output_model"
    batch_size = 8
    learning_rate = 3e-5
    num_epochs = 3

    data_module = MLMDataModule(data_path=data_path, batch_size=batch_size)
    data_module.setup() 

    model = BERTMLM(tokenizer=data_module.tokenizer, learning_rate=learning_rate, num_training_steps=num_epochs * len(data_module.train_dataloader()))

    checkpoint_callback = ModelCheckpoint(dirpath=output_dir, monitor="train_loss", save_top_k=1, mode="min")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(model, datamodule=data_module)
