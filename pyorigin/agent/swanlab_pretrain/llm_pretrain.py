import datasets
import transformers
import swanlab
from swanlab.integration.huggingface import SwanLabCallback
import modelscope

def main():
    swanlab.init("WikiLLM")

    print("\033[0;31;40m++加载原始数据中·····················\033[0m")
    raw_datasets = datasets.load_dataset(
        "json", data_files="WIKI_CN/wikipedia-zh-cn-20241020.json"
    )

    raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=2333)
    print('/n')
    print("dataset info")
    print(raw_datasets)

    # 下载config.json文件
    modelscope.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B").save_pretrained(
        "Qwen2-0.5B"
    )
    # 下载tokenizer.json文件
    """
    tokenizer.json：分词器的配置文件，包含了词汇表和分词器所需的其它元数据。
    merges.txt：用于Byte-Pair Encoding (BPE)分词的合并规则文件（如果分词器是基于BPE的话）。
    vocab.txt：分词器的词汇表文件（如果分词器是基于词汇表的话）。
    special_tokens_map.json：特殊令牌的映射文件，比如[CLS]，[SEP]等。
    tokenizer_config.json：分词器的配置信息。

    还有added_tokens.jsom文件
    """
    modelscope.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B").save_pretrained(
        "Qwen2-0.5B"
    )
    print("\033[0;31;40m++++++++++++++++++原始数据加载完毕++++++++++++++++++++\033[0m"+"\n")






    print("\033[0;31;40m++处理原始数据中·····················\033[0m")
    context_length = 512
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "./Qwen2-0.5B"
    )
    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    # 把源数据都进行了处理，处理完之后得到tokens，处理函数就是上面那个函数。
    print("tokenize dataset info")
    print(tokenized_datasets)
    print("\033[0;31;40m+++++++++++++++++处理数据完毕++++++++++++++++++++\033[0m"+"\n")








    print("\033[0;31;40m++模型初始化构建中·····················\033[0m")
    # 结束令牌作为填充令牌
    tokenizer.pad_token = tokenizer.eos_token
    # 不使用掩码，仅仅只是用普通语言模型
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

    config = transformers.AutoConfig.from_pretrained(
        "./Qwen2-0.5B",
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        num_hidden_layers=12,
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # 使用上面的配置初始化了一个因果语言模型
    model = transformers.Qwen2ForCausalLM(config)
    model_size = sum(t.numel() for t in model.parameters())
    print("Model Config:")
    print(config)
    print(f"Model Size: {model_size/1000**2:.1f}M parameters")
    print("\033[0;31;40m+++++++++++++++++模型初始化完毕++++++++++++++++++++\033[0m"+"\n")







    print("\033[0;31;40m++训练参数配置中·····················\033[0m")
    # 训练参数
    args = transformers.TrainingArguments(
        output_dir="WikiLLM",
        per_device_train_batch_size=13,  # 每个GPU的训练batch数
        per_device_eval_batch_size=13,  # 每个GPU的测试batch数
        eval_strategy="steps",
        eval_steps=5_00,
        logging_steps=50,
        gradient_accumulation_steps=8,  # 梯度累计总数
        num_train_epochs=2,  # 训练epoch数
        weight_decay=0.1,
        warmup_steps=2_00,
        optim="adamw_torch",  # 优化器使用adamw
        lr_scheduler_type="cosine",  # 学习率衰减策略
        learning_rate=5e-4,  # 基础学习率，
        save_steps=5_00,
        save_total_limit=10,
        bf16=True,  # 开启bf16训练, 对于Amper架构以下的显卡建议替换为fp16=True
    )
    print("Train Args:")
    print(args)
    print("\033[0;31;40m+++++++++++++++++++++++++++++++++++训练参数配置完毕++\033[0m"+"\n")










    print("\033[0;31;40m++模型训练中·····················\033[0m")
    # enjoy training
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        callbacks=[SwanLabCallback()],
    )
    trainer.train()
    print("\033[0;31;40m+++++++++++++++++训练完毕++++++++++++++++++++\033[0m"+"\n")









    print("\033[0;31;40m++模型生成任务中·····················\033[0m")
    model.save_pretrained("./WikiLLM/Weight")
    pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("GENERATE:", pipe("人工智能", num_return_sequences=1)[0]["generated_text"])
    prompts = ["牛顿", "北京市", "亚洲历史"]
    examples = []
    for i in range(3):
        text = pipe(prompts[i], num_return_sequences=1)[0]["generated_text"]
        text = swanlab.Text(text)
        examples.append(text)
    swanlab.log({"Generate": examples})
    print("\033[0;31;40m+++++++++++++++++++++++++++++++++++生成任务完毕++\033[0m"+"\n")


if __name__ == "__main__":
    main()
