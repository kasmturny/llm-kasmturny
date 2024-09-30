import os
from typing import List
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

class OpenAIBgeEmbeddings:
    def __init__(self):
        model_name = "./config/libs/bge-large-zh-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.model = HuggingFaceBgeEmbeddings(
            model_name=os.path.join(os.path.dirname(__file__), model_name),
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def get_embedding(self, query: str) -> List[float]:
        vector = self.model.embed_query(query)
        return vector

if __name__ == '__main__':
    test = OpenAIBgeEmbeddings().get_embedding("你好")
    print(test)
