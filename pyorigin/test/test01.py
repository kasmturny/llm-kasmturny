import pandas as pd
from tqdm import tqdm


test_jsonl_new_path = "D:\\Exploitation\\All\\llm-kasmturny\\pyorigin\\agent\\bio_ner_finetune\\data\\test.jsonl"


test_total_df = pd.read_json(test_jsonl_new_path, lines=True)
test_total_df_len = len(test_total_df)
test_df = test_total_df[0:test_total_df_len]
# 假设test_df是你的DataFrame
for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    # 在这里处理每一行
    print(index)