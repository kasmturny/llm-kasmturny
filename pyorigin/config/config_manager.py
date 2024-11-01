"""
主要就是把config.ini转化为一个对象好方便调用
"""
import os
import configparser


class ModelConfig:
    def __init__(self, model_name, api_key, base_url, langchain_api_key, langchain_tracing_v2):
        self.model_name: str = model_name
        self.api_key: str = api_key
        self.base_url: str = base_url
        self.langchain_api_key: str = langchain_api_key
        self.langchain_tracing_v2: bool = langchain_tracing_v2
class RedisConfig:
    def __init__(self, host, port, db,password):
        self.host: str = host
        self.port: str = port
        self.db: str = db
        self.password: str = password
class MilvusConfig:
    def __init__(self, host, port, username, password):
        self.host: str = host
        self.port: str = port
        self.username: str = username
        self.password: str = password

class LocalEmbeddingConfig:
    def __init__(self, local_embedding_path):
        self.local_embedding_path: str = local_embedding_path

class EmbeddingConfig:
    def __init__(self, api_key,embedding_url, embedding_model_name):
        self.api_key : str = api_key
        self.embedding_url: str = embedding_url
        self.embedding_model_name: str = embedding_model_name

class KafkaConfig:
    def __init__(self, host,port, group_id, topic):
        self.host: str = host
        self.port: str = port
        self.group_id: str = group_id
        self.topic: str = topic

class BertConfig:
    def __init__(self, base_model_name,wwm_model_name):
        self.base_model_name: str = base_model_name
        self.wwm_model_name: str = wwm_model_name


class ConfigManager:
    def __init__(self):
        self.config = configparser.ConfigParser()
        path = os.path.join(os.path.dirname(__file__), '../config.ini')
        self.config.read(path)

    def get_model_config(self) -> ModelConfig:
        """大模型配置:"""
        model_name = self.config.get("MODEL", "MODEL_NAME", fallback="gpt-3.5-turbo")
        api_key = self.config.get("MODEL", "API_KEY")
        base_url = self.config.get("MODEL", "BASE_URL")
        langchain_api_key = self.config.get("MODEL", "LANGCHAIN_API_KEY")
        langchain_tracing_v2 = self.config.get("MODEL", "LANGCHAIN_TRACING_V2").lower()
        return ModelConfig(model_name, api_key, base_url, langchain_api_key, langchain_tracing_v2)

    def get_redis_config(self) -> RedisConfig:
        """redis"""
        host = self.config.get("REDIS", "HOST")
        port = self.config.get("REDIS", "PORT")
        db = self.config.get("REDIS", "DB")
        password = self.config.get("REDIS", "PASSWORD")
        return RedisConfig(host, port, db, password)

    def get_milvus_config(self) -> MilvusConfig:
        """向量数据库"""
        host = self.config.get("MILVUS", "HOST")
        port = self.config.get("MILVUS", "PORT")
        username = self.config.get("MILVUS", "USERNAME")
        password = self.config.get("MILVUS", "PASSWORD")
        return MilvusConfig(host, port, username, password)

    def get_local_embedding_config(self) -> LocalEmbeddingConfig:
        """本地embedding路径"""
        local_embedding_path = self.config.get("LOCAL_EMBEDDING", "EMBEDDING_PATH")
        return LocalEmbeddingConfig(local_embedding_path)

    def get_embedding_config(self) -> EmbeddingConfig:
        """embedding配置"""
        api_key = self.config.get("EMBEDDING", "API_KEY")
        embedding_url = self.config.get("EMBEDDING", "EMBEDDING_URL")
        embedding_model_name = self.config.get("EMBEDDING", "EMBEDDING_MODEL_NAME")
        return EmbeddingConfig(api_key, embedding_url, embedding_model_name)

    def get_kafka_config(self) -> KafkaConfig:
        """kafka配置"""
        host = self.config.get("KAFKA", "HOST")
        port = self.config.get("KAFKA", "PORT")
        group_id = self.config.get("KAFKA","GROUP_ID")
        topic = self.config.get("KAFKA","TOPIC")
        return KafkaConfig(host, port, group_id,topic)

    def get_bert_config(self) -> BertConfig:
        """bert配置"""
        base_model_name = self.config.get("BERT", "BASE_MODEL_NAME")
        wwm_model_name = self.config.get("BERT", "WWM_MODEL_NAME")
        return BertConfig(base_model_name, wwm_model_name)


if __name__ == "__main__":
    # model = ConfigManager().get_model_config()
    # redis = ConfigManager().get_redis_config()
    # milvus = ConfigManager().get_milvus_config()
    # local_embedding = ConfigManager().get_local_embedding_config()
    # embeding = ConfigManager().get_embedding_config()
    # kafka = ConfigManager().get_kafka_config()
    bert = ConfigManager().get_bert_config()
    print('断点')
