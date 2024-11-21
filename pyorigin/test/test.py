def convert_data_to_entities_format(data_list):
    # 初始化最终的实体列表
    entities_list = []

    # 遍历每个数据项
    for data in data_list:
        text = data['text']
        labels = data['label']

        # 遍历每个实体类型
        for entity_type, entities in labels.items():
            # 遍历每个实体
            for entity_name, spans in entities.items():
                # 遍历每个实体出现的范围
                for span in spans:
                    # 获取实体文本
                    entity_text = text[span[0]:span[1]]
                    # 添加到实体列表中
                    entities_list.append({
                        "entity_type": entity_type,
                        "entity_text": entity_text
                    })

    # 返回转换后的实体列表
    return {"entities": entities_list}


# 示例使用
data = [
    {
        "text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
        "label": {
            "name": {
                "叶老桂": [[9, 11], [32, 34]]
            },
            "company": {
                "浙商银行": [[0, 3]]
            }
        }
    }
    # 可以在这里继续添加更多的数据项
]

# 调用函数并打印结果
converted_data = convert_data_to_entities_format(data)
print(converted_data)
