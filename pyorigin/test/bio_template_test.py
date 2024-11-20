def bio_template_test():
    import json

    import numpy as np
    from tqdm import tqdm

    from pyorigin.core.base_agent import BigModel
    import time
    start = time.time()
    data = np.load('D:\\Exploitation\\All\\llm-kasmturny\\pyorigin\\agent\\bio_ner_finetune\\data\\test.npz',
                   allow_pickle=True)
    input_list = []
    output_list = []
    input_text = []
    for i in tqdm(range(len(data['words']))):
        word = data['words'][i]
        label = data['labels'][i]
        input_list.append([])
        output_list.append([])
        input_text.append(''.join(word))
        for j in range(len(word)):
            input_list[i].append([word[j], j])
            output_list[i].append([word[j], j, label[j]])
    end = time.time()
    print(end - start)

    model = BigModel()
    output = model.str_output_invoke(
        content={'input_list': input_list[1], 'input_text': input_text[1]},
        template="""Role:
        你是一个命名实体识别的专家，可以很好的识别文本中的实体信息，实体信息有十类：地址、书籍、公司、游戏、政府、电影、名称、组织、职位、场景
    Parameters:
        input_list:{input_list}
        input_text:{input_text}
    Workflow:
        1. 仔细分析input_text，结合语境，分析出文中的实体信息
        2. 根据input_list和input_text的对应关系，按照实体信息和标记规则对input_list进行标注
        3. 将标注结果设置为output_list,并按照输出格式进行进行输出，输出格式必须是json格式
    Rule:
        1. 实体信息：['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
        2. 标注标签：["O", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name', 'B-organization', 'B-position', 'B-scene', "I-address", "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name', 'I-organization', 'I-position', 'I-scene', "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie', 'S-name', 'S-organization', 'S-position', 'S-scene']
        3. 标注规则：'O'用来标注非实体，如果实体长度大于1，则用'B-label'标记实体开始，用'I-label'标记实体中间和结尾，如果实体长度为1，则用'S-label'标记实体
    Example:
        input:{{
                "input_text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，"
                "input_list": "[['彭', 0], ['小', 1], ['军', 2], ['认', 3], ['为', 4], ['，', 5], ['国', 6], ['内', 7], ['银', 8], ['行', 9], ['现', 10], ['在', 11], ['走', 12], ['的', 13], ['是', 14], ['台', 15], ['湾', 16], ['的', 17], ['发', 18], ['卡', 19], ['模', 20], ['式', 21], ['，', 22], ['先', 23], ['通', 24], ['过', 25], ['跑', 26], ['马', 27], ['圈', 28], ['地', 29], ['再', 30], ['在', 31], ['圈', 32], ['的', 33], ['地', 34], ['里', 35], ['面', 36], ['选', 37], ['择', 38], ['客', 39], ['户', 40], ['，', 41]]"
                }}
        output:{{
                "output_list": "[['彭', 0, 'B-name'], ['小', 1, 'I-name'], ['军', 2, 'I-name'], ['认', 3, 'O'], ['为', 4, 'O'], ['，', 5, 'O'], ['国', 6, 'O'], ['内', 7, 'O'], ['银', 8, 'O'], ['行', 9, 'O'], ['现', 10, 'O'], ['在', 11, 'O'], ['走', 12, 'O'], ['的', 13, 'O'], ['是', 14, 'O'], ['台', 15, 'B-address'], ['湾', 16, 'I-address'], ['的', 17, 'O'], ['发', 18, 'O'], ['卡', 19, 'O'], ['模', 20, 'O'], ['式', 21, 'O'], ['，', 22, 'O'], ['先', 23, 'O'], ['通', 24, 'O'], ['过', 25, 'O'], ['跑', 26, 'O'], ['马', 27, 'O'], ['圈', 28, 'O'], ['地', 29, 'O'], ['再', 30, 'O'], ['在', 31, 'O'], ['圈', 32, 'O'], ['的', 33, 'O'], ['地', 34, 'O'], ['里', 35, 'O'], ['面', 36, 'O'], ['选', 37, 'O'], ['择', 38, 'O'], ['客', 39, 'O'], ['户', 40, 'O'], ['，', 41, 'O']]"
                }}
    Output:
        {{
            "output_list":output_list
        }}""",
    )

    output_dict = json.loads(output)
    print(output)

def tqdm_test():
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

if __name__ == '__main__':
    bio_template_test()
    # tqdm_test()
    print('test')
