一、项目地址
https://github.com/Zeyi-Lin/LLM-Finetune

二、这个只能在GPU上面跑

三、训练
```
cd /
conda create -p /kasmturny/venv
conda activate /kasmturny/venv
pip install torch swanlab modelscope transformers datasets peft accelerate pandas tiktoken
cd ./kasmturny
git clone https://github.com/kasmturny/llm-kasmturny.git
cd ./llm-kasmturny/pyorigin/agent/ner_finetune
python /kasmturny/llm-kasmturny/pyorigin/agent/ner_finetune/ner_finetune.py
```