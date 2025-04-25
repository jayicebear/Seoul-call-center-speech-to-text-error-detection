from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments

model_name = "google/mt5-small"  # 또는 "mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Load correction dataset
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = load_dataset("json", data_files={"train": "correction_data.jsonl"}, split="train")
tokenized_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./mt5_corrector",
    evaluation_strategy="no",
    per_device_train_batch_size=8,
    num_train_epochs=5,
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
