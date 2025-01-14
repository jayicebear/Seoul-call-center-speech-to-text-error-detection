import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'
file_path = r'/home/jayicebear/snap/Untitled Folder/데이터_v10.xlsx'
df_pred = pd.read_excel(file_path, sheet_name='인식문')
df_true = pd.read_excel(file_path, sheet_name='정답문')

df_pred.drop(['in_out'], axis=1, inplace=True)
df_true.drop(['in_out'], axis=1, inplace=True)

df_total = df_pred.assign(정답문=df_true['정답문'])
dialogues = df_total['KEY'].unique()


train_df, df_temp = train_test_split(df_total, test_size=0.3, random_state=42)

val_df, test_df = train_test_split(df_temp, test_size=2/3, random_state=42)

import pandas as pd
# Assuming 'df' is your DataFrame
# Separate matching and non-matching samples
test_df_match = test_df[test_df['STT인식문'] == test_df['정답문']]
test_df_nonmatch = test_df[test_df['STT인식문'] != test_df['정답문']]
# Desired sample sizes
n_total = 2000
n_match = int(n_total * 0.9)    # 90% matching
n_nonmatch = n_total - n_match  # 10% non-matching
# Adjust sample sizes if not enough data is available
n_match = min(n_match, len(test_df_match))
n_nonmatch = min(n_nonmatch, len(test_df_nonmatch))
# Resample to meet the total number of samples
if n_match + n_nonmatch < n_total:
    n_match = min(len(test_df_match), n_total - n_nonmatch)
    n_nonmatch = min(len(test_df_nonmatch), n_total - n_match)
# Randomly sample from each subset
sample_match = test_df_match.sample(n=n_match, random_state=42)
sample_nonmatch = test_df_nonmatch.sample(n=n_nonmatch, random_state=42)
# Combine and shuffle the final sample
test_df = pd.concat([sample_match, sample_nonmatch]).sample(frac=1, random_state=42).reset_index(drop=True)
# Now 'final_sample' contains your desired 2000 samples

correctness = []
for index, row in test_df.iterrows():
        if row['STT인식문'] == row['정답문']:
            correctness.append(0)
        else:
            correctness.append(1)

Model_name = ['yanolja_eeve']
#base_model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
for i in range(len(Model_name)):
    model_dir = f'/home/jayicebear/snap/Seoul/Results/V10/{Model_name[i]}'
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    outputs4 = []
    num = 0
    for index, row in test_df.iterrows():
        _input = row["STT인식문"]
        prompt = '''제시된 발화에 이상이 없다면 "이상없음", 이상이 있다면 틀린 부분만을 반환하세요.\n'''
        prompt += f"입력: {_input}\n출력:" 

        messages = [
            {"role": "system",
             "content": ""
            },
            {"role": "user", "content": prompt
             },
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        output = model.generate(
            input_ids.to("cuda"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            do_sample=False
        )

        print(tokenizer.decode(output[0]))
        outputs4.append([tokenizer.decode(output[0]), index])
        print('\n\n')

        
def evaluate(Model_name):
    if Model_name == 'eeve':
    
    
    if Model_name == 'Llama3.1':
    
    
    if Model_name == 'Qwen':
error_index = []
_len = 0
correct = 0
for exam in outputs4:
    _len += 1
    correction = exam[0].split("assistant")[-1].split("<|im_end|>")[0].strip()
    ori_input = exam[0].split("입력:")[-1].split("출력:")[0].strip()
    print(correction)
    print(ori_input)
    if ori_input == correction or "이상없음" in correction:
        error_index.append(0)
        correct += 1
        continue
    error_index.append(1)
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model_name', dest='Model_name', type=str)
    args = parser.parse_args()

    Model_name = args.Model_name
