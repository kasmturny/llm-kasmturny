# 数据下载
https://huggingface.co/datasets/fjcanyue/wikipedia-zh-cn

# 内容结构

```bash
llm_pretrain/Qwen2-0.5B
总计 16M
-rw-r--r-- 1 root root   80 12月  5 18:51 added_tokens.json
-rw-r--r-- 1 root root  760 12月  5 18:51 config.json
-rw-r--r-- 1 root root 1.6M 12月  5 18:51 merges.txt
-rw-r--r-- 1 root root  370 12月  5 18:51 special_tokens_map.json
-rw-r--r-- 1 root root 1.3K 12月  5 18:51 tokenizer_config.json
-rw-r--r-- 1 root root  11M 12月  5 18:51 tokenizer.json
-rw-r--r-- 1 root root 2.7M 12月  5 18:51 vocab.json
```

```bash
llm_pretrain/WikiLLM/checkpoint-9892
总计 1.4G
-rw-r--r-- 1 root root   80 12月  6 05:46 added_tokens.json
-rw-r--r-- 1 root root  690 12月  6 05:46 config.json
-rw-r--r-- 1 root root   95 12月  6 05:46 generation_config.json
-rw-r--r-- 1 root root 1.6M 12月  6 05:46 merges.txt
-rw-r--r-- 1 root root 471M 12月  6 05:46 model.safetensors
-rw-r--r-- 1 root root 941M 12月  6 05:46 optimizer.pt
-rw-r--r-- 1 root root  14K 12月  6 05:46 rng_state.pth
-rw-r--r-- 1 root root 1.1K 12月  6 05:46 scheduler.pt
-rw-r--r-- 1 root root  256 12月  6 05:46 special_tokens_map.json
-rw-r--r-- 1 root root 1.3K 12月  6 05:46 tokenizer_config.json
-rw-r--r-- 1 root root  11M 12月  6 05:46 tokenizer.json
-rw-r--r-- 1 root root  39K 12月  6 05:46 trainer_state.json
-rw-r--r-- 1 root root 5.2K 12月  6 05:46 training_args.bin
-rw-r--r-- 1 root root 2.7M 12月  6 05:46 vocab.json
```

```bash
llm_pretrain/WikiLLM/Weight
总计 471M
-rw-r--r-- 1 root root  690 12月  6 05:46 config.json
-rw-r--r-- 1 root root   95 12月  6 05:46 generation_config.json
-rw-r--r-- 1 root root 471M 12月  6 05:46 model.safetensors
```

# 代码逻辑

——加载数据

——处理数据

——下载tokenier文件然后加载tokeniner

——下载模型config文件然后初始化一个模型

——训练参数配置

——开始训练

——训练的时候会生成一堆的checkpoint然后最后生成一个Weight

——利用生成的Weight加上一个tokeniner就可以完成一个模型的使用