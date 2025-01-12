import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import T5TokenizerFast, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json
import torch
from difflib import SequenceMatcher

device = 'cuda' if torch.cuda.is_available() else 'cpu'

file_path = r'/home/jayicebear/snap/Untitled Folder/'
df_pred = pd.read_excel(file_path, sheet_name='인식문')
df_true = pd.read_excel(file_path, sheet_name='정답문')

df_pred.drop(['in_out'], axis=1, inplace=True)
df_true.drop(['in_out'], axis=1, inplace=True)

df_total = df_pred.assign(정답문=df_true['정답문'])


data = df_total

from_to = []
for i in range(len(data)):
    input_str = data.iloc[i]['STT인식문']
    output_str = data.iloc[i]['정답문']

    matcher = SequenceMatcher(None, input_str, output_str)
    diffs = ''

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # tag는 'equal', 'replace', 'delete', 'insert' 중 하나
        if tag == 'replace':
            changed_from = input_str[i1:i2]
            changed_to = output_str[j1:j2]
            diffs += f"({changed_from} -> {changed_to}) "
    if not diffs:
        diffs += '이상없음'
    
    from_to.append(diffs)

# 리스트를 새로운 컬럼으로 할당
df_total['변경부분'] = from_to

dialogues = df_total['KEY'].unique()


train_df, df_temp = train_test_split(df_total, test_size=0.3, random_state=42)

val_df, test_df = train_test_split(df_temp, test_size=2/3, random_state=42)
#'google/mt5-base',MT5Tokenizer,MT5ForConditionalGeneration,
model_name_list = ['google/mt5-base','paust/pko-t5-base','gogamza/kobart-base-v2']
model_tokenizer_list = [MT5Tokenizer,T5TokenizerFast,PreTrainedTokenizerFast]
model_load_list = [MT5ForConditionalGeneration,T5ForConditionalGeneration,BartForConditionalGeneration]



# 1. 데이터 로드 및 정제
# 이미 분할된 데이터셋 (train_df, val_df, test_df)이 있다고 가정합니다.
# 각 데이터프레임에는 'input_text'와 'target_text' 컬럼이 있어야 합니다.

# 2. Dataset 객체로 변환
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# 3. 토크나이저 및 모델 로드
for i in range(len(model_name_list)):
    model_name = model_name_list[i]
    tokenizer = model_tokenizer_list[i].from_pretrained(model_name)
    model = model_load_list[i].from_pretrained(model_name)

    # 4. 데이터셋 토크나이징
    max_input_length = 128
    max_target_length = 128

    def preprocess_function(examples):
        inputs = examples['STT인식문']
        targets = examples['변경부분']
        model_inputs = tokenizer(
            inputs, max_length=max_input_length, truncation=True, padding='max_length'
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, truncation=True, padding='max_length'
            )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

    # 5. compute_metrics 함수 정의
    def build_compute_metrics(eval_dataset):
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred

            # 디코딩
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # 공백 제거
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            # 입력 텍스트 가져오기
            input_texts = eval_dataset['STT인식문']

            # 레이블 생성
            true_labels = [0 if input_text == label else 1 for input_text, label in zip(input_texts, decoded_labels)]

            # 예측값 생성
            pred_labels = [0 if pred == input_text else 1 for pred, input_text in zip(decoded_preds, input_texts)]

            # f1-score 계산
            f1 = f1_score(true_labels, pred_labels)
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels)
            recall = recall_score(true_labels, pred_labels)

            return {
                'f1': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }
        return compute_metrics

    compute_metrics_fn = build_compute_metrics(eval_dataset)

    # 6. 트레이너 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir='Results/results',
        num_train_epochs=15,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=100,
        learning_rate=1e-4,
        weight_decay=1e-3,
        save_total_limit=2,
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )

    # 7. 모델 학습
    trainer.train()

    # 8. 모델 저장
    trainer.save_model(f'Results/V13_changed_{model_name}')
