
"""
微调数据：
message = {
    instrusction: "请根据以下文本，标注出其中的实体，并给出实体类型。"
    input: "文本原文"
    output: "{
        "entities": [
            {
                "entity_type": "实体类型",
                "entity_text": "实体文本",
            }，
        ]
    }"
}
"""
def convert_to_json(origin_path, output_path=None):
    with open(origin_path, 'r', encoding='utf-8') as f:
        origin_data = f.readlines()
        messages = []
        for line in origin_data:
            data = json.loads(line)
            input_text = data['text']
            entities = []
            for key1, value1 in data['label'].items():
                entity_type = key1
                for key2, value2 in value1.items():
                    entity_text = key2
                    entities.append({"entity_text": entity_text,"entity_type": entity_type})
            output = {}
            output['entities'] = entities
            output = json.dumps(output, ensure_ascii=False)

            instruction = """你是一个命名实体识别的专家，请对文本中的 地址（address）、书籍（book）、公司（company）、游戏（game）、政府（government）、电影（movie
            ）、名称（name）、组织（organization）、职位（position）、场景（scene） 
            这十个实体进行识别，对识别到的实体选取输入原文中连续的字符文本进行表示，并确定实体类型，输出json表示的识别结果。例如输入input
            :"浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，"，那么输出output:"{"entities":[{"entity_text": "叶老桂",
            "entity_type": "name"},{"entity_text": "浙商银行","entity_type": "company"}]}"。 1、注意输出一定要是json格式的字符串. 
            2、entity_text的值必须是原文中的连续的文本，不能随意截取拼接原文中的文本,也不要进行原文文本的补全和标点符号的补全. 
            3、entity_type的值必须是这十个实体类型中的一个，不能随意添加其他实体类型."""



            message = {
                "instruction": instruction,
                "input": input_text,
                "output": output
            }
            messages.append(message)

    # with open(output_path, "w", encoding="utf-8") as file:
    #     for message in messages:
    #         file.write(json.dumps(message, ensure_ascii=False) + "\n")

    return messages


"""
输入："文本原文"
||||||||
经过大模型
||||||||
输出：
"{
    "entities": [
        {
            "entity_type": "实体类型",
            "entity_text": "实体文本",
        }，
    ]
}"
检查措施：
    输出必须是json字符串
    只有一个字段entities,字段的值为列表
    列表中的每个元素是一个字典，字典只有两个字段：entity_type和entity_text
    entity_type和entity_text的值必须是字符串
    entity_type的值必须是实体类型列表中的值
    entity_text的值必须是原文中的连续的文本
"""


import json

def llm_output_check(llm_output, origin_text,
                     entity_types=['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene'])-> bool:
    """
    检查LLM输出是否符合要求
    :param llm_output:
    :param origin_text:
    :param entity_types:
    :return:
    """
    try:
        # 尝试将输出解析为JSON
        data = json.loads(llm_output)

        # 检查是否只有一个字段"entities"，并且其值是列表
        if not isinstance(data, dict) or set(data.keys()) != {"entities"} or not isinstance(data["entities"], list):
            print("错误：输出必须只包含一个字段'entities'，其值为列表。")
            return False

        # 遍历列表中的每个元素
        for entity in data["entities"]:
            # 检查每个元素是否是字典，并且只有"entity_type"和"entity_text"两个键
            if not isinstance(entity, dict) or set(entity.keys()) != {"entity_type", "entity_text"}:
                print("错误：'entities'中的每个元素必须是只包含'entity_type'和'entity_text'键的字典。")
                return False

            # 检查"entity_type"和"entity_text"的值是否为字符串
            if not isinstance(entity["entity_type"], str) or not isinstance(entity["entity_text"], str):
                print("错误：'entity_type'和'entity_text'必须为字符串。")
                return False

            # 检查"entity_type"的值是否在有效的实体类型列表中
            if entity["entity_type"] not in entity_types:
                print(f"错误：'entity_type' '{entity['entity_type']}'不在有效的实体类型列表中。")
                return False

            # 检查"entity_text"的值是否在原文中连续出现
            if entity["entity_text"] not in origin_text:
                print(f"错误：'entity_text' '{entity['entity_text']}'不是原文中的连续文本。")
                return False

        # 如果所有检查都通过，打印成功消息并返回True
        return True

    except json.JSONDecodeError as e:
        # 如果输出不是有效的JSON，打印错误并返回False
        print(f"错误：输出不是有效的JSON。{e}")
        return False


"""
输入："文本原文"
输出：
"{
    "entities": [
        {
            "entity_type": "实体类型",
            "entity_text": "实体文本",
        }，
    ]
}"
"""
def output_data_bio(llm_output,origin_text):


    llm_output = json.loads(llm_output)
    origin_text_list = list(origin_text)
    BIO_list = [None for _ in range(len(origin_text_list))]

    for entity in llm_output['entities']:
        entity_text = entity['entity_text']

        for i in range(len(origin_text_list)-len(entity_text)+1):
            if origin_text_list[i:i+len(entity_text)] == list(entity_text):
                if len(entity_text) == 1:
                    BIO_list[i] = 'S-' + entity['entity_type']
                else:
                    BIO_list[i] = 'B-' + entity['entity_type']
                    for j in range(1, len(entity_text)):
                        BIO_list[i+j] = 'I-' + entity['entity_type']
        for i in range(len(origin_text_list)):
            if BIO_list[i] is None:
                BIO_list[i] = 'O'
    return BIO_list







if __name__ == "__main__":
    """匹配得到的BIO数据和原来的数据有265/10748条是不匹配的，原因是有可能多个实体的话，仅仅只有第一个实体进行了标注"""
    # messages = convert_to_json('../agent/check_ner_finetune/data/train.json')
    # BIO_list = []
    # for message in messages:
    #     input = message['input']
    #     output = message['output']
    #     bio = output_data_bio(output, input)
    #     BIO_list.append(bio)
    # import numpy as np
    # data = np.load('../agent/check_ner_finetune/data/train.npz', allow_pickle=True)
    # labels = list(data['labels'])
    # words = data['words']
    # error_nums = 0
    # for i in range(len(BIO_list)):
    #     if BIO_list[i] == labels[i]:
    #         pass
    #     else:
    #         print(f"第{i}条数据错误")
    #         error_nums += 1
    # print(f"错误条数：{error_nums}")

    llm_output = """{"entities":[{"entity_text": "叶", "entity_type": "name"}, {"entity_text": "浙商银行", "entity_type": "company"}]}"""
    origin_text = '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，'
    print(llm_output_check(llm_output, origin_text))
    print(output_data_bio(llm_output,origin_text))
    """这样标注的一个bio会和原本的有点不一样"""
    # ['B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-organization', 'I-organization', 'O','B-organization', 'I-organization', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','B-name', 'I-name', 'I-name', 'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    # ['B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-organization', 'I-organization', 'O', 'B-organization', 'I-organization', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-organization', 'I-organization', 'O', 'O', 'B-name', 'I-name', 'I-name', 'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    # {
    #     "text": "金石开：建议关注一下做客维冈的米堡吧。联赛初期表现不错的米堡因为索斯盖特拿了一个8月最佳教练奖，",
    #     "label": {
    #         "organization": {
    #             "维冈": [
    #                 [
    #                     12,
    #                     13
    #                 ]
    #             ],
    #             "米堡": [
    #                 [
    #                     15,
    #                     16
    #                 ]
    #             ]
    #         },
    #         "name": {
    #             "金石开": [
    #                 [
    #                     0,
    #                     2
    #                 ]
    #             ],
    #             "索斯盖特": [
    #                 [
    #                     32,
    #                     35
    #                 ]
    #             ]
    #         }
    #     }
    # }
    pass



