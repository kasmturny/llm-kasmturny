from pymilvus import MilvusClient
from config.config_manager import ConfigManager
import os
from langchain_openai import ChatOpenAI
import redis


class InitServer:
    def __init__(self):
        self.config = ConfigManager()

    def get_model(self) -> ChatOpenAI:
        model_config = self.config.get_model_config()
        os.environ["OPENAI_API_KEY"] = model_config.api_key
        os.environ["OPENAI_BASE_URL"] = model_config.base_url
        # os.environ["LANGCHAIN_TRACING_V2"] = model_config.langchain_tracing_v2
        # os.environ["LANGCHAIN_API_KEY"] = model_config.langchain_api_key
        model_name = model_config.model_name
        model = ChatOpenAI(temperature=0, model=model_name)
        return model

    def get_redis(self) -> redis.Redis:
        redis_config = self.config.get_redis_config()
        redis_server = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password
        )
        return redis_server

    def get_milvus(self) -> MilvusClient:
        milvus_config = self.config.get_milvus_config()
        client = MilvusClient(
            uri=f"http://{milvus_config.host}:{milvus_config.port}"
        )
        return client

if __name__ == "__main__":
    model = InitServer().get_model()
    milvus = InitServer().get_milvus()
    redis = InitServer().get_redis()
    redis.rpush("kiana",'我老婆' )
    print('断点')

