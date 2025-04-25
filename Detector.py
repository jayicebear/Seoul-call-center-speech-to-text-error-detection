from transformers import ElectraTokenizer, ElectraForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch

# Load tokenizer and model
model_name = "monologg/koelectra-base-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(model_name)
model = ElectraForTokenClassification.from_pretrained(model_name, num_labels=2)

# Load detection dataset
def tokenize_and_align_labels(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
    word_ids = tokenized.word_ids()

    labels = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != prev_word_idx:
            labels.append(example["labels"][word_idx])
        else:
            labels.append(-100)
        prev_word_idx = word_idx
    tokenized["labels"] = labels
    return tokenized

dataset = load_dataset("json", data_files={"train": "detection_data.jsonl"}, split="train")
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)

training_args = TrainingArguments(
    output_dir="./koelectra_token_cls",
    evaluation_strategy="no",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
