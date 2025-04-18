# model e.g.: meta-llama/Llama-2-7b-hf

gpuid=0
k_bits=2
v_bits=2
group_size=32
residual_length=32
model=meta-llama/Llama-3.2-1B-Instruct
e=1

CUDA_VISIBLE_DEVICES=$gpuid python pred_long_bench.py --model_name_or_path $model \
    --cache_dir ./cached_models \
    --k_bits $k_bits \
    --v_bits $v_bits \
    --group_size $group_size \
    --residual_length $residual_length \
    --e ${e}

#python eval_long_bench.py --model Llama-3.2-1B-Instruct_2048_2bits_group32_residual128