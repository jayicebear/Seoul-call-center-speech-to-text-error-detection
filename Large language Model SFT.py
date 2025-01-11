import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig, EarlyStoppingCallback
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import find_all_linear_names, print_trainable_parameters
from huggingface_hub import login

login(token='')

model_name = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
checkpoint = ["final_checkpoint_Llama", "final_checkpoint_Qwen"]
#model_name = ["sh2orc/Llama-3.1-Korean-8B-Instruct","yanolja/EEVE-Korean-Instruct-10.8B-v1.0"]
#checkpoint = ["korean_llama3.1","yanolja_eeve"]
data_files = {
    'train': '/home/jayicebear/snap/Seoul/train_dataset/train_V13.json',
    'validation': '/home/jayicebear/snap/Seoul/train_dataset/val_V13.json'
}

# 일단은 binary classification을 목적으로 튜닝을 시켜보자. 그 후에 이상이 있다고 판단한 문장들만 대해서 튜닝을 진행해 보자.
# Bert 기반의 모델을 사용하는 것도 괜찮을 것 같다.
# T5 기반의 모델로 사용하여 보자.
# ChatGPT로 레이블을 다시 한번 가공하고, 바뀐 레이블로 시행하여 보자.

def formatting_prompts_func(example):
    output_texts = []
        
    instruction = '''제시된 발화에 이상이 없다면 발화를 그대로 반환하고, 이상이 있다면 발화를 고쳐서 반환하세요.'''
    for i in range(len(example['input'])):
        text = f"{instruction}\n\n입력: {example['input'][i]}\n출력: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

for i in range(2):
    output_dir="/home/jayicebear/snap/Seoul/Results/V13/"
    datasets = load_dataset("json", data_files=data_files)
    train_dataset = datasets['train']
    eval_dataset = datasets['validation']

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(model_name[i], torch_dtype=torch.bfloat16, quantization_config=bnb_config) #, device_map = 'auto')                                                
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    tokenizer = AutoTokenizer.from_pretrained(model_name[i])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # Change the LORA hyperparameters accordingly to fit your use case
    peft_config = LoraConfig(
            r=32,
            lora_alpha=8,
            target_modules=find_all_linear_names(base_model),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
    )

    base_model = get_peft_model(base_model, peft_config)
    print_trainable_parameters(base_model)
    
    # TrainingArguments에 evaluation 전략과 early stopping 관련 설정 추가
    training_args = TrainingArguments(
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            num_train_epochs=5, 
            learning_rate=2e-5,
            bf16=True,
            save_total_limit=5,
            logging_steps=10,
            output_dir=output_dir,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_strategy="epoch",
    )

    trainer = SFTTrainer(
            model=base_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=256,
            formatting_func=formatting_prompts_func,
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train() 
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, checkpoint[i])
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.cuda.empty_cache()         
