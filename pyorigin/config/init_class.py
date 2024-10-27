from pyorigin.config.config_manager import ConfigManager

from pymilvus import MilvusClient
import os
from langchain_openai import ChatOpenAI
import redis
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

from pyorigin.config.config_manager import Embedding


class InitClass:
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

    def get_embedding(self) -> Embedding:
        return self.config.get_embedding_config()



if __name__ == "__main__":
    # model = InitClass().get_model()
    # redis = InitClass().get_redis()
    # milvus = InitClass().get_milvus()
    # local_embedding = InitClass().get_local_embedding()
    embeding = InitClass().get_embedding()
    print('断点')


