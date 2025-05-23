import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import transformers
os.environ["WANDB_DISABLED"] = "true"

from utils.process_args import process_args
from transformers import LlamaConfig, MistralConfig, AutoTokenizer


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # For results in KIVI paper (Llama, Llama-Chat, Mistral-7B-v0.1), we do not apply any special treatment to the prompt.
    # For lmsys/longchat-7b-v1.5-32k and mistralai/Mistral-7B-Instruct-v0.2, we need to rewrite the prompt a little bit.
    # Update: we add the template for the new llama-3-instruct model
    if "llama-3" in model_name.lower() and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "longchat" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral-v0.2-instruct" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    preds = []
    # f = open("out.txt", "w")
    # i = 0
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # i += 1
        # f.write("##############################\n")
        # f.write(f"{i} prompt:\n{prompt}\n")
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        # if(len(prompt) > max_length):
        #     half = int(max_length / 2)
        #     prompt = prompt[:half] + prompt[-half:]
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        # f.write(f"############################################\n{i} prompt:\n{prompt}")
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def get_pred_batch(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, batch_size=1):
    preds = []
    data = data.select(range(50))
    # 针对samsum任务的特殊性，可能需要逐条处理
    if dataset == "samsum":
        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = (tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                          tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True))
            inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = inputs.input_ids.shape[-1]
            output = model.generate(
                **inputs,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_name)
            preds.append({
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"]
            })
    else:
        # 针对其他任务，使用批量处理
        # print(type(data))
        tokenizer.pad_token = tokenizer.eos_token
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data.select(range(i, i + batch_size))
            prompts = []
            for json_obj in batch:
                prompt = prompt_format.format(**json_obj)
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = (tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                              tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True))
                if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    prompt = build_chat(tokenizer, prompt, model_name)
                prompts.append(prompt)
            # 批量tokenize，注意设置padding确保各样本对齐
            inputs = tokenizer(prompts, truncation=False, padding=True, return_tensors="pt").to(device)
            # 获取每个样本的实际长度（非padding部分）
            context_lengths = inputs.attention_mask.sum(dim=1).tolist()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )
            # 对每个样本，从对应的 context_length 处开始解码
            for j, output in enumerate(outputs):
                pred = tokenizer.decode(output[context_lengths[j]:], skip_special_tokens=True)
                pred = post_process(pred, model_name)
                preds.append({
                    "pred": pred,
                    "answers": batch[j]["answers"],
                    "all_classes": batch[j]["all_classes"],
                    "length": batch[j]["length"]
                })
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    # args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model_name = args.model

    # define your model
    model_args, data_args, training_args = process_args()
    # print(model_args, data_args, training_args)
    model_name = model_args.model_name_or_path.split("/")[-1]
    # dtype = torch.bfloat16 if training_args.bf16 else torch.float
    dtype = torch.float16
    
    if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
        config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True)
                                            # model_max_length=training_args.model_max_length)
    elif 'mistral' in model_args.model_name_or_path.lower():
        config = MistralConfig.from_pretrained(model_args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, 
                                            use_fast=False, 
                                            trust_remote_code=True)
    else:
        raise NotImplementedError
    
    if 'llama' in model_args.model_name_or_path.lower() or 'longchat' in model_args.model_name_or_path.lower():
        if model_args.k_bits < 16 and model_args.v_bits < 16:
            from models.llama_kivi import LlamaForCausalLM_KIVI
            config.k_bits = model_args.k_bits
            config.v_bits = model_args.v_bits
            config.group_size = model_args.group_size
            config.residual_length = model_args.residual_length
            config.use_flash = True # Note: We activate the flashattention to speed up the inference
            print("used llamaForCausalLM_KIVI")
            model = LlamaForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_flash_attention_2=True,
                device_map="auto",
            )
        else:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_flash_attention_2=True,
                device_map="auto",
            )

    elif 'mistral' in model_args.model_name_or_path.lower():
        if model_args.k_bits < 16 and model_args.v_bits < 16:
            from models.mistral_kivi import MistralForCausalLM_KIVI
            config.k_bits = model_args.k_bits
            config.v_bits = model_args.v_bits
            config.group_size = model_args.group_size
            config.residual_length = model_args.residual_length
            config.use_flash = False
            model = MistralForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            from transformers import MistralForCausalLM
            model = MistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_flash_attention_2=True,
                device_map="auto",
            )

    else:
        raise NotImplementedError

    #
    # Load model directly
    # tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
    # model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K")

    model.eval()
    max_length = model2maxlen[model_name]
    if data_args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["triviaqa"]
        # datasets = ["triviaqa", "qasper", "trec", "samsum", "lcc", "repobench-p", "qmsum", "multi_news"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        if data_args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_residual{model_args.residual_length}"):
                os.makedirs(f"pred_e/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_residual{model_args.residual_length}")
            out_path = f"pred_e/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_residual{model_args.residual_length}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_residual{model_args.residual_length}"):
                os.makedirs(f"pred/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_residual{model_args.residual_length}")
            out_path = f"pred/{model_name}_{max_length}_{model_args.k_bits}bits_group{model_args.group_size}_residual{model_args.residual_length}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred_batch(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
        torch.cuda.synchronize()
        with open("memory.txt", "w", encoding="utf-8") as f:
            f.write(f"{dataset}'s peak memory: {torch.cuda.max_memory_allocated() / 1024 ** 3}GB\n")