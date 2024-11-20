一、项目地址
https://github.com/Zeyi-Lin/LLM-Finetune

二、这个只能在GPU上面跑

三、训练
```
cd /
conda create -p /hy-tmp/venv
conda activate /hy-tmp/venv
pip install torch swanlab modelscope transformers datasets peft accelerate pandas tiktoken
cd ./hy-tmp
git clone https://github.com/kasmturny/llm-kasmturny.git
cd ./llm-kasmturny/pyorigin/agent/ner_finetune
python /hy-tmp/llm-kasmturny/pyorigin/agent/ner_finetune/qwen_train.py
python /hy-tmp/llm-kasmturny/pyorigin/agent/ner_finetune/glm_train.py
```