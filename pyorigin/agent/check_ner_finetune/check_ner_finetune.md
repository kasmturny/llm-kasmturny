一、命令行
```
cd /
conda create -p /hy-tmp/kasmturny/venv
conda activate /hy-tmp/kasmturny/venv
pip install torch swanlab modelscope transformers datasets peft accelerate pandas tiktoken tqdm
cd ./hy-tmp/kasmturny
git clone https://github.com/kasmturny/llm-kasmturny.git
cd ./llm-kasmturny/pyorigin/agent/check_ner_finetune
python /hy-tmp/kasmturny/llm-kasmturny/pyorigin/agent/check_ner_finetune/check_ner_finetune.py
```