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
BATCH_SIZE = 5
PATH_TO_YOUR_SAVE_DIR = './cached_models'

model_name_or_path = 'meta-llama/Llama-3.2-1B-Instruct'
config = LlamaConfig.from_pretrained(model_name_or_path)
config.k_bits = K_BITS # current support 2/4 bit for KV Cache
config.v_bits = V_BITS # current support 2/4 bit for KV Cache
config.group_size = GROUP_SIZE
config.use_flash = True
config.residual_length = RESIDUAL_LENGTH # the number of recent fp16 tokens
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

if K_BITS < 16 and V_BITS < 16:
    print('use kivi')
    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=PATH_TO_YOUR_SAVE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
        device_map="auto",
    )
else:
    print('use llama')
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        config=config,
        cache_dir=PATH_TO_YOUR_SAVE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
        device_map="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, 
    use_fast=False, 
    trust_remote_code=True)

model.cuda().eval()

context = []
batch_size = BATCH_SIZE
prompt_lenth = 2048
output_length = 10240
num_repeats = 1
for _ in range(batch_size):
    string = 't,' * (prompt_lenth // 2)
    context.append(string[:-1])
inputs = tokenizer(context, return_tensors="pt").to('cuda')
input_ids = inputs['input_ids']
print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\nmodel:{model_name_or_path}\nkbits:{K_BITS}\n")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    torch.cuda.synchronize()
    st = time.time()
    for i in range(num_repeats):
        outputs = model.generate(**inputs, max_new_tokens=output_length)
    torch.cuda.synchronize()
    print(f'used time: {(time.time() - st) / num_repeats * 1000} ms')
    used_mem = torch.cuda.max_memory_allocated()
    print(f'peak mem: {used_mem / 1024 ** 3} GB')
    print(torch.cuda.mem_get_info())
# print(torch.cuda.memory_stats())

# 32 ： 160 + 320 : 507629568
# 32 : 160 + 320 :  232902656

# batch_size: 2 rs: 128
# 2 + SCOPE： 1024 + 4096 : (942387200, 8585216000)    used time: 100479.40683364868 ms
# 2 : 1024 + 4096: (652517376, 8585216000) used time: 114294.39067840576 ms
# 32: 1024 + 4096: (942321664, 8585216000) used time: 102109.39812660217 ms
# batch_size: 3 rs: 128
# 2 + SCOPE： (2110783488, 8585216000) 1024 + 4096 :  used time: 110633.80479812622 ms
# 2 : 1024 + 4096: (1583734784, 8585216000) used time: 115432.38139152527 ms
# 32: 1024 + 4096: (2171600896, 8585216000) used time: 98614.34173583984 ms
# batch_size: 4 rs: 128
# 2 + SCOPE： 1024 + 4096 : (942387200, 8585216000)   used time: 100479.40683364868 ms
# 2 : 1024 + 4096: (652517376, 8585216000) used time: 114294.39067840576 ms
# 32: 1024 + 4096: (884645888, 8585216000) used time: 104770.42269706726 ms
# batch_size: 5 rs: 128
# 2 + SCOPE： 1024 + 4096 :  (30961664, 8585216000)  112783.358335495 ms
# 2 : 1024 + 4096: (0, 8585216000) used time: 140176.66935920715 ms
# 32: 1024 + 4096: (96419840, 8585216000) used time: 143557.95216560364 ms
# batch_size: 6 rs: 128
# 2 + SCOPE： 1024 + 4096 :  (0, 8585216000)  113309.44752693176 ms
# 2 : 1024 + 4096: (0, 8585216000) used time: 123242.88892745972 ms
# 32: 1024 + 4096: (0, 8585216000) used time: 120247.31540679932 ms


# batch_size: 5 rs: 128
# 2 + SCOPE： 1024 + 4096 :  (30961664, 8585216000)  112783.358335495 ms
# 2 : 1024 + 4096: (0, 8585216000) used time: 140176.66935920715 ms
# 32: 1024 + 4096: (96419840, 8585216000) used time: 143557.95216560364 ms

# batch_size: 5 rs: 128
# 2 + SCOPE： 1024 + 5120 :(116785152, 8585216000)   145815.29307365417 ms
# 2 : 1024 + 5120:  (0, 8585216000) used time:  160946.91157341003 ms
# 32: 1024 + 5120: (0, 8585216000) used time: 156660.3524684906 ms

# batch_size: 5 rs: 128
# 2: 1025+6144: (16293888, 8585216000)used time:176102.99563407898 ms
# 32：1025+6144: (16293888, 8585216000) used time: 176102.99563407898 ms
# 2 + SCOPE: 1025+6144: (176959488, 8585216000) used time: 177278.89680862427 ms

# batch_size: 5 rs: 128
# 2: 1025+7168: (0, 8585216000) used time: 239107.67221450806 ms
# 32：1025+7168: (40558592, 8585216000) used time: 261866.83011054993 ms
# 2 + SCOPE: 1025+7168: (163852288, 8585216000) used time: 201831.25233650208 ms

# batch_size: 5 rs: 128
# 2: 1025+8192: (0, 8585216000)  used time: 239107.67221450806 ms
# 32：1025+8192: (0, 8585216000) used time: 291164.048910141 ms
# 2 + SCOPE: 1025+8192: (204390400, 8585216000) used time: 226997.01476097107 ms

# batch_size: 5 rs: 128
# 2: 1025+9216: (0, 8585216000)  used time: 334472.81098365784 ms
# 32：1025+9192: (52076544, 8585216000) used time: 471851.22776031494 ms
# 2 + SCOPE: 1024 + 9192: (144719872, 8585216000) used time: 247878.89695167542 ms


# batch_size: 7 rs: 128

# batch_size: 30 rs: 128
# 2: 65 + 2048 (228560896, 8585216000) used time: 96573.73046875 ms
# 32: 65 + 2048 (0, 8585216000) used time: 119182.36064910889 ms
# 2 + SCOPE：(1422594048, 8585216000) used time: 88709.72275733948 ms
# batch_size: 25 rs: 128
# 2: 65 + 2048 (758476800, 8585216000) used time: 88570.25742530823 ms
# 32: 65 + 2048  (0, 8585216000) used time: 98346.98987007141 ms
# 2 + SCOPE： (2297430016, 8585216000) used time:  86483.90436172485 ms
# batch_size: 20 rs: 128
# 2: 65 + 2048 (1286639616, 8585216000) used time: 79073.23956489563 ms
# 32: 65 + 2048 (235245568, 8585216000) used time: 64934.97443199158 ms
# 2 + SCOPE (2754609152, 8585216000)： used time:  72779.88076210022 ms
# batch_size: 15 rs: 128
# 2: 65 + 2048  (2479882240, 8585216000) used time: 67832.09586143494 ms
# 32: 65 + 2048  (1260236800, 8585216000) used time: 59650.14576911926 ms
# 2 + SCOPE： (3239051264, 8585216000) used time:   66893.31698417664 ms
# batch_size: 10 rs: 128
# 2: 65 + 2048 (3232759808, 8585216000) used time:  used time: 72100.05283355713 ms
# 32: 65 + 2048 (3077570560, 8585216000) used time: 55450.18148422241 ms
# 2 + SCOPE： (3729784832, 8585216000) used time:  61681.45990371704 ms



# batch_size: 30 rs: 32
# 2: 65 + 2048 (519745536, 8585216000) used time: 92815.34028053284 ms
# 32: 65 + 2048 (0, 8585216000) used time: 133863.13581466675 ms
# 2 + SCOPE：(1655963648, 8585216000) used time: 88125.3399848938 ms