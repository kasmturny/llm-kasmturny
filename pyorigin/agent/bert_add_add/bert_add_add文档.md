# 一、新建虚拟环境
# 二、安装依赖
```
pip install langchain==0.3.4 langchain-core==0.3.11 langchain-community==0.3.3 langchain-openai==0.2.3 langgraph==0.2.39

pip install pymilvus==2.4.8 redis==5.2.0 kafka-python==2.0.2

pip install fastapi==0.115.4 uvicorn==0.32.0 streamlit==1.39.0

pip install retry==0.9.2 tqdm==4.66.5

pip install sentence_transformers==3.2.1 tensorflow==2.18.0 tf-keras==2.18 pytorch-crf==0.7.2 torch==2.5.1 transformers==4.46.2
```
# 三、修改代码

第一个：bert-crf

首先你的huggingface/hub下面要有一个全新的bert-base-chinese,然后执行下面的操作

model.py的依赖要更新一下
data_loader的模型要改一下
run.py的模型要改一下       
config.py的gpu改为cpu
train.py的模型要改一下
在文件下创建./case/bad_case.txt,并在metrics之中改文件编码utf-8
run.py前面加上西面这一行
```
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
```

第二个：bert-lstm-crf

主要是transfromers==2.2.2的问题
```
pip install transformers==2.2.2
```
然后在目录下面放入两个模型，一个是bert-base-chinese,一个是roberta-wwm-large,确保是全新的
config.py的gpu改为cpu
在文件下创建./case/bad_case.txt,并在metrics之中改文件编码utf-8
run.py前面加上西面这一行
```
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
```
可以在run.py里面指定到底要训练bert-base-chinese还是roberta-wwm-large,两个模型二选一，base模型的f1有点低，不知道为什么

第三个：bilstm-crf

config.py的gpu变为cpu
metrics.py的文件编码改一下utf-8，还有就是新建bad_case.txt

第四个：bert-softmax

更新model.py的依赖
更新run.py的模型还有【环境变量】
更新data_loader.py的模型
config.py的gpu改为cpu
train.py的模型改一下
在文件下创建./case/bad_case.txt,并在metrics之中改文件编码utf-8
```
"""
注意config的引用问题
"""
```
# 四、项目地址
本文档使用后面的项目地址进行部署，项目地址：https://github.com/hemingkx/CLUENER2020












