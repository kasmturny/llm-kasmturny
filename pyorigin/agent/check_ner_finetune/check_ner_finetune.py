
"""
微调数据：
instrusction: "请根据以下文本，标注出其中的实体，并给出实体类型。"
input: "文本原文"
output: {
    "entities": [
        {
            "entity_type": "实体类型",
            "entity_text": "实体文本",
        }，
    ]
}
"""

"""
输入：文本原文
||||||||
经过大模型
||||||||
输出：
{
    "entities": [
        {
            "entity_type": "实体类型",
            "entity_text": "实体文本",
        }，
    ]
}
检查措施：
    输出必须是json字符串
    只有一个字段entities,字段的值为列表
    列表中的每个元素是一个字典，字典只有两个字段：entity_type和entity_text
    entity_type和entity_text的值必须是字符串
    entity_type的值必须是实体类型列表中的值
    entity_text的值必须是原文中的连续的文本
"""


import json

def llm_output_check(llm_output, origin_text, entity_types)-> bool:
    """
    检查LLM输出是否符合要求
    :param llm_output:
    :param origin_text:
    :param entity_types:['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    :return:
    """
    try:
        # 尝试将输出解析为JSON
        data = json.loads(llm_output)

        # 检查是否只有一个字段"entities"，并且其值是列表
        if not isinstance(data, dict) or "entities" not in data or not isinstance(data["entities"], list):
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
        print("所有检查通过。")
        return True

    except json.JSONDecodeError as e:
        # 如果输出不是有效的JSON，打印错误并返回False
        print(f"错误：输出不是有效的JSON。{e}")
        return False


if __name__ == "__main__":
    # 示例使用
    text = "Alice went to the store."  # 示例文本原文
    valid_entity_types = ['Person', 'Location', 'Organization']  # 示例实体类型列表
    output = '{"entities": [{"entity_type": "Person", "entity_text": "Alice"}]}'  # 示例输出

    # 调用函数
    llm_output_check(output, text, valid_entity_types)



