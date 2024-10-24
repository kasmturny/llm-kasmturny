from pymilvus import MilvusClient
from pyorigin.config.config_manager import ConfigManager
import os
from langchain_openai import ChatOpenAI
import redis
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

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

    def get_local_embedding(self) -> HuggingFaceBgeEmbeddings:
        model_name = self.config.get_local_embedding_config().local_embedding_path
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embedding_model= HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return embedding_model

if __name__ == "__main__":
    test = InitServer().get_local_embedding()
    print(test.embed_query("你好"))

