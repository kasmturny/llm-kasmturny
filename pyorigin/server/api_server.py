import datetime

from pyorigin.config.init_class import InitClass
from pyorigin.core.base_agent import BigModel, LocalEmbedding, Embedding, Redis, Milvus, Kafka

from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel



# 一、#######################前期准备###########################
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 二、#######################数据结构##########################
class Chat(BaseModel):
    question: str

class ChatHistory:
    def __init__(self):
        self.messages = []

    def add_message(self, question: str, response: str):
        self.messages.append({'question': question, 'response': response})
# 实例化，用于存储聊天消息
chat_history = ChatHistory()

# 三、#################路由测试################################

@app.post("/chat")
async def process_statement(statement: Chat):
    content = {"input": statement.question, "history": chat_history.messages}
    template = """
    你不是机器人，你是一只可爱的兔子，你的名字叫兔憨憨
    向你提问的人是你的主人，你应该在每一句回答之前加上'狗修金撒嘛，'几个字，语气要可爱
    当前时间为：%s。\n历史记录：{history}\n用户输入：{input}
    """% (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    response = BigModel().str_output_invoke(
        content=content,
        template=template
    )
    chat_history.add_message(statement.question, response)
    return {"response":response}

# 四、#################启动类################################
class fastapi_run:
    def __init__(self):
        self.app = app
        self.run()

    def run(self):
        uvicorn.run(app=self.app, host="0.0.0.0", port=8000, workers=1)


if __name__ == '__main__':
    # uvicorn.run意味着uvicorn启动了一个轻量级的web服务器，这个服务器监听0.0.0.0:8000,实际上就是监听127.0.0.1:8000，或者是localhost:8000
    # 本机浏览器——127.0.0.1:8000——服务器
    # 反正就是你浏览去输入了127.0.0.1:8000//kasmturny，你就请求了服务器上的/kasmturny路径，希望从这个路径获取资源
    # app="fastapi01:app"意味着uvicorn会查找名为fastapi01的模块，并从中获取名为app的FastAPI实例。
    # 首先就是请求127.0.0.1:8000//kasmturny，然后进入uvicorn请求资源，但是uvicorn运行着fastapi01的app实例，这个实例返回json
    fastapi_run()