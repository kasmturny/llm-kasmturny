import hashlib
import json
import time


from pymilvus import DataType, FieldSchema, CollectionSchema
from typing import List, Dict, Any, Union
from retry import retry
from pyorigin.config.init_class import InitClass
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser

class BigModel:
    def __init__(self):
        self.model = InitClass().get_model()

    @retry(tries=3)
    def str_output_invoke(self, content, template, examples="", output_parser=StrOutputParser(), history=""):
        model = self.model
        prompt = PromptTemplate.from_template(
            str(template)
        ).partial(examples=examples, history=history)
        output_text = (prompt | model | output_parser).invoke(input=content)
        return output_text

    @retry(tries=3)
    def json_output_invoke(self, content, template,output_parser=JsonOutputParser(), examples="", history=""):
        model = self.model
        prompt = PromptTemplate.from_template(
            str(template)
        ).partial(examples=examples, history=history)
        output_text = (prompt | model | output_parser).invoke(input=content)
        return output_text

class Redis:
    def __init__(self):
        self.redis = InitClass().get_redis()

    def push_redis(self,data: dict, group_id: str):
        redis = self.redis
        data.update({"chat_time": round(time.time())})  # 添加时间戳字段
        key = f"chat_server:{group_id}"  # 在chat_server:后面的group_id作为键
        redis.expire(key, 24 * 60 * 60 * 7)  # 过期时间7天
        redis.rpush(key, json.dumps(data, ensure_ascii=False))

class Milvus:
    def __init__(self):
        self.milvus = InitClass().get_milvus()
        self.embedding = InitClass().get_local_embedding()

    def create_milvus_collection(self, collection_name: str, fields: List[FieldSchema] = None, description: str = "默认无描述"):
        """没有fields就使用默认参数，集合的描述默认为None"""
        if fields is None:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True,max_length=32, description="primary id"),
                FieldSchema(name="input", dtype=DataType.VARCHAR, max_length=65535, description="input"),
                FieldSchema(name="output", dtype=DataType.VARCHAR, max_length=65535, description="output"),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024, description="vector")
            ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True, description=description)
        self.milvus.create_collection(collection_name=collection_name, schema=schema)
        print(f"集合 {collection_name} 已创建.")

    def delete_milvus_collection(self, collection_name: str):
        """删除集合"""
        self.milvus.drop_collection(collection_name)
        print(f"集合 {collection_name} 已删除.")

    def insert_milvus_data(self, collection_name: str, data: List[Dict[str, str]], is_deduplication=True):
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

    def query_milvus_data(self, query: str, collection_name: str, vector_size: int, output_fields: List[str] = None, limit:int=1):
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

if __name__ == "__main__":

