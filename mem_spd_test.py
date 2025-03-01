# LLaMA model with KIVI
import torch
import os
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
import time

torch.cuda.empty_cache()
K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128
BATCH_SIZE = 1
PATH_TO_YOUR_SAVE_DIR = './cached_models'

model_name_or_path = 'meta-llama/Llama-3.2-1B-Instruct'
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_bits = K_BITS # current support 2/4 bit for KV Cache
config.v_bits = V_BITS # current support 2/4 bit for KV Cache
config.group_size = GROUP_SIZE
config.use_flash = False
config.residual_length = RESIDUAL_LENGTH # the number of recent fp16 tokens
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

if K_BITS < 16 and V_BITS < 16:
    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
else:
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True)

model.cuda().eval()

context = []
batch_size = BATCH_SIZE
prompt_lenth = 90
output_length = 3000
num_repeats = 1
for _ in range(batch_size):
    string = 't,' * (prompt_lenth // 2)
    context.append(string[:-1])
for idx, value in enumerate(context):
    context[idx] = "output 3000 words"

inputs = tokenizer(context, return_tensors="pt").to('cuda')
input_ids = inputs['input_ids']
print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\nmodel:{model_name_or_path}")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    torch.cuda.synchronize()
    st = time.time()
    for i in range(num_repeats):
        outputs = model.generate(input_ids, max_new_tokens=output_length)
    torch.cuda.synchronize()
    print(f'used time: {(time.time() - st) / num_repeats * 1000} ms')
    used_mem = torch.cuda.max_memory_allocated()
    print(f'peak mem: {used_mem / 1024 ** 3} GB')