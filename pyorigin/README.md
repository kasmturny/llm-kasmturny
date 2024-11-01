# llm-kasmturny
这是一个llm仓库
## 一、pip 依赖
```
pip install langchain langchain-core langchain-community langchain-openai langgraph

pip install pymilvus redis kafka-python

pip install sentence_transformers

pip install fastapi uvicorn

pip install streamlit

pip install retry

pip install tensorflow tf-keras
```
## 二、命令行
```
python -m streamlit run D:\Exploitation\All\llm-kasmturny\pyorigin\server\chat_streamlit_server.py
```
## 三、tensor转化pytorch
```
python D:\Exploitation\All\llm-kasmturny\pyorigin\utils\tensor_to_pytorch_util.py --tf_checkpoint_path D:/Exploitation/All/llm-kasmturny/model/chinese_L-12_H-768_A-12/bert_model.ckpt.index --bert_config_file D:/Exploitation/All/llm-kasmturny/model/chinese_L-12_H-768_A-12/bert_config.json  --pytorch_dump_path  D:/Exploitation/All/llm-kasmturny/model/chinese_L-12_H-768_A-12/pytorch_model.bin

python D:\Exploitation\All\llm-kasmturny\pyorigin\utils\tensor_to_pytorch_util.py --tf_checkpoint_path D:/Exploitation/All/llm-kasmturny/model/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt.index --bert_config_file D:/Exploitation/All/llm-kasmturny/model/chinese_wwm_L-12_H-768_A-12/bert_config.json  --pytorch_dump_path  D:/Exploitation/All/llm-kasmturny/model/chinese_wwm_L-12_H-768_A-12/pytorch_model.bin
```