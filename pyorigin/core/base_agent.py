import hashlib
import json
import re
import time

import requests
from pymilvus import DataType, FieldSchema, CollectionSchema, Collection
from typing import List, Dict, Any, Union, Callable
from retry import retry
from pyorigin.config.init_class import InitClass
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from kafka.errors import KafkaError
from kafka import KafkaProducer, KafkaConsumer

class BigModel:
    def __init__(self):
        self.model = InitClass().get_model()

    @retry(tries=3)
    def str_output_invoke(self, content, template, examples="", output_parser=StrOutputParser(), history="") -> str:
        model = self.model
        prompt = PromptTemplate.from_template(
            str(template)
        ).partial(examples=examples, history=history)
        output_text = (prompt | model | output_parser).invoke(input=content)
        return output_text

    @retry(tries=3)
    def json_output_invoke(self, content, template,output_parser=JsonOutputParser(), examples="", history="") -> Any:
        model = self.model
        prompt = PromptTemplate.from_template(
            str(template)
        ).partial(examples=examples, history=history)
        output_text = (prompt | model | output_parser).invoke(input=content)
        return output_text

class LocalEmbedding:
    def __init__(self):
        self.embedding = InitClass().get_local_embedding()

    def get_embedding(self, text: str)-> List[float]:
        return self.embedding.embed_query(text)

class Embedding:
    def __init__(self):
        self.embedding = InitClass().get_embedding()

    def get_embedding(self, text: str) -> List[float]:
        method="POST"
        url=self.embedding.embedding_url
        headers={'Authorization': f'Bearer {self.embedding.api_key}',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Content-Type': 'application/json'
        }
        data=json.dumps({"model": self.embedding.embedding_model_name,
                        "input": text
                        })
        response = requests.request(method=method, url=url, headers=headers, data=data)
        return response.json()['data'][0]["embedding"]

class Redis:
    def __init__(self):
        self.redis = InitClass().get_redis()

    def push_redis(self,data: dict, group_id: str):
        redis = self.redis
        data.update({"chat_time": round(time.time())})  # 添加时间戳字段
        key = f"chat_server:{group_id}"  # 在chat_server:后面的group_id作为键
        redis.expire(key, 24 * 60 * 60 * 7)  # 过期时间7天
        redis.rpush(key, json.dumps(data, ensure_ascii=False))

    def query_redis(self, group_id: str, start: int = 0, end: int = -1) -> List[str]:
        redis = self.redis
        key = f"chat_server:{group_id}"
        messages = redis.lrange(key, start, end)
        return [json.loads(message) for message in messages]

    def clear_redis(self, group_id: str,):
        redis = self.redis
        key = f"chat_server:{group_id}"
        redis.delete(key)

class Milvus:
    def __init__(self):
        self.milvus = InitClass().get_milvus()
        self.embedding = InitClass().get_local_embedding()

    def create_milvus_collection(self, collection_name: str, fields: List[FieldSchema] = None, description: str = "默认无描述"):
        """没有fields就使用默认参数，集合的描述默认为None"""
        if self.milvus.has_collection(collection_name):
            print(f"集合 {collection_name} 已存在,不可重复创建.")
            return
        if fields is None:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id", auto_id=True),
                FieldSchema(name="md5id", dtype=DataType.VARCHAR, max_length=32,description="md5id"),
                FieldSchema(name="input", dtype=DataType.VARCHAR, max_length=65535, description="input"),
                FieldSchema(name="output", dtype=DataType.VARCHAR, max_length=65535, description="output"),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024, description="vector")
            ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True, description=description)
        self.milvus.create_collection(collection_name=collection_name, schema=schema)
        index_params = self.milvus.prepare_index_params()
        index_params.add_index(
            field_name="embeddings",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 1024}
        )
        self.milvus.create_index(collection_name=collection_name, index_params=index_params)
        print(f"集合 {collection_name} 已创建.")

    def delete_milvus_collection(self, collection_name: str):
        """删除集合"""
        self.milvus.drop_collection(collection_name)
        print(f"集合 {collection_name} 已删除.")

    def insert_milvus_data(self, collection_name: str, data: List[Dict[str, str]], is_deduplication=False):
        # 集合名字collection_name: str, 数据是对应集合的字典列表data: List[Dict[str, str]]
        if is_deduplication:
            res = self.milvus.upsert(
                # 去重
                collection_name=collection_name,
                data=data
            )
        else:
            res = self.milvus.insert(
                # 不去重
                collection_name=collection_name,
                data=data
            )
        print(res)

    def query_milvus_data(self, query: str, collection_name: str, vector_size: int, output_fields: List[str] = None, limit:int=1) -> Any:
        """查询数据"""
        vector = self.embedding.embed_query(query)
        res = self.milvus.search(
            collection_name=collection_name,
            data=[vector],  # 查询向量，是一个列表
            limit=limit,  # 搜索结果最大数量
            search_params={
                "metric_type": "L2",
                "params": {"nprobe": vector_size},
            },
            output_fields=output_fields
        )
        return res

class Kafka:
    def __init__(self):
        self.kafka = InitClass().get_kafka()
        self.bootstrap_servers = [self.kafka.host + ":" + str(self.kafka.port)]

    def produce(self, topic, message):
        producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers,
                                 value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                                 api_version=(0, 10, 1))
        try:
            producer.send(topic, value=message)
            producer.flush()
            print(f"成功发送数据{message}到{topic}")
        except KafkaError as e:
            print(f"发送数据失败: {e}")
        finally:
            producer.close()

    def consume(self, topic, group_id, func: Callable, *args, **kwargs):
        consumer = KafkaConsumer(bootstrap_servers=self.bootstrap_servers,
                                 group_id=group_id,
                                 auto_offset_reset='earliest',
                                 enable_auto_commit=False,
                                 api_version=(0, 10, 2))
        consumer.subscribe([topic])
        for message in consumer:
            message_dict: dict = eval(re.sub(r"\s+", " ", message.value.decode('utf-8')))
            func(message_dict, *args, **kwargs)
            print(f"（Group_Id：{group_id}）从（Topic:{topic}）收到消息: {message.value},已处理,已提交")



if __name__ == "__main__":
    """Model测试"""
    # print(BigModel().str_output_invoke("你好","你是一个小兔子，请回答{content}"))
    """milvus测试"""
    # Milvus().delete_milvus_collection("rabbit")
    # Milvus().create_milvus_collection("rabbit")
    # templates = [
    #     {"input": "兔子最可爱", "output": "兔子最可爱"},
    #     {"input": "兔子最漂亮", "output": "兔子最漂亮"},
    #     {"input": "兔子最美丽", "output": "兔子最美丽"},
    #     {"input": "兔子最温柔", "output": "兔子最温柔"},
    #     {"input": "兔子最善良", "output": "兔子最善良"},
    #     {"input": "兔子最聪明", "output": "兔子最聪明"},
    #     ]
    # datas = []
    # for template in templates:
    #     template['md5id'] = hashlib.md5(template['input'].encode('utf-8')).hexdigest()
    #     template['embeddings'] = InitClass().get_local_embedding().embed_query(template['input'])
    #     print(template['input'])
    #     datas.append(template)
    # Milvus().insert_milvus_data("rabbit", datas)
    # print(Milvus().query_milvus_data("兔子脑袋瓜最好",
    #                                  "rabbit",
    #                                  1024,
    #                                  output_fields=["input", "output"]))
    """redis测试"""
    # Redis().push_redis({'name':'兔子','chat':'我是兔子'}, "114514")
    # Redis().push_redis({'name': 'kasmturny', 'chat': '我喜欢麻辣兔头'}, "114514")
    # print(Redis().query_redis(group_id="114514"))
    """local_embedding测试"""
    # print(LocalEmbedding().get_embedding("兔子最可爱"))
    """embedding测试"""
    # print(Embedding().get_embedding("兔子最可爱"))
    """kafka测试"""
    # massages = [
    #     {"key": "key0", "value": 0},
    #     {"key": "key1", "value": 1},
    #     {"key": "key2", "value": 2},
    #     {"key": "key3", "value": 3},
    #     {"key": "key4", "value": 4},
    #     {"key": "key5", "value": 5},
    #     {"key": "key6", "value": 6},
    #     {"key": "key7", "value": 7},
    #     {"key": "key8", "value": 8},
    #     {"key": "key9", "value": 9}
    #
    # ]
    # for message in massages:
    #     Kafka().produce("test", message)
    # def print_add_hundred(message, hun):
    #     print(message['value']+hun)
    # Kafka().consume("test", "kasmturny", print_add_hundred ,100)
    print("断点")