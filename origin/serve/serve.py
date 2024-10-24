from enum import Enum
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from origin.core.init_server import InitServer


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

# 三、#################路由测试################################

@app.post("/chat")
async def process_statement(statement: Chat):
    llm = InitServer().get_model()
    response = llm.invoke(statement.question).content
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