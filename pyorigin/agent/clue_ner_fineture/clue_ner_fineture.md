一、pip环境
```
多了一个tqdm
```
二、命令行
```
cd /
conda create -p /kasmturny/venv
conda activate /kasmturny/venv
pip install torch swanlab modelscope transformers datasets peft accelerate pandas tiktoken tqdm
cd ./kasmturny
git clone https://github.com/kasmturny/llm-kasmturny.git
cd ./llm-kasmturny/pyorigin/agent/clue_ner_fineture
python /kasmturny/llm-kasmturny/pyorigin/agent/ner_finetune/qwen_train_clue2020.py
```
