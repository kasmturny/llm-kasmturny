一、命令行
```
cd /
conda create -p /kasmturny/venv
conda activate /kasmturny/venv
pip install torch swanlab modelscope transformers datasets peft accelerate pandas tiktoken tqdm
cd ./kasmturny
git clone https://github.com/kasmturny/llm-kasmturny.git
cd ./llm-kasmturny/pyorigin/agent/bio_ner_finetune
python /kasmturny/llm-kasmturny/pyorigin/agent/bio_ner_finetune/bio_qwen_train.py
```
二、运行时间：10+小时