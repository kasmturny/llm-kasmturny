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
二、运行时间：24+小时
```
反正，f1值极差，看来又要重新设计了
```
三、需要多卡就把环境变量的注释取消掉