import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def restore_kv(expanded_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, expanded_size, slen, head_dim = expanded_states.shape
    assert expanded_size % n_rep == 0, "Expanded size does not match expected size."
    restored_states = expanded_states.view(batch, expanded_size // n_rep, n_rep, slen, head_dim)
    restored_states = restored_states.mean(dim=2)  # 或者使用其他聚合方法
    return restored_states


def get_repeat_tensor(x: torch.Tensor, g: int):
    """
    [x1, x2, x3, ... , xn] to 
    [x1 * g, x1 * g + 1, x1 * g + 2, ..., x1 * g + g - 1,
     x2 * g, x2 * g + 1, x2 * g + 2, ..., x2 * g + g - 1,
     ...
    xn * g, xn * g + 1, xn * g + 2, ..., xn * g + g - 1]
    """
    indices = torch.arange(g)
    expanded_indices = indices.repeat(x.shape[-1], 1).to(x.device)
    # print('expanded_indices', expanded_indices)
    # 计算结果
    result = x.unsqueeze(-1) * g + expanded_indices
    # print('result', result)
    # 展平结果张量
    res = torch.flatten(result, start_dim=-2)
    
    return res

def P(o):
    if o is not None:
        print(o)
    else:
        print("None") 

def P_S(o):
    if o is not None:
        print(o.shape)
    else:
        print("Nones")
class ALLKVCluster():
    
    allkv_max_capacity_prompt = 0
    current_decoding_step = 0
    jump_step = 0

    def __init__(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, prefill_window_size = 2048, prefill_recent_size = 8):
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric # None, fixed, linear, jump
        self.decoding_window_size = decoding_window_size # b1 + b2
        self.decoding_recent_size = decoding_recent_size # b2
        self.prefill_window_size = prefill_window_size # a1 + a2
        self.prefill_recent_size = prefill_recent_size # a2
        assert self.decoding_window_size - self.decoding_recent_size > 0
        assert self.prefill_window_size - self.prefill_recent_size > 0            

    def reset(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, prefill_window_size = 2048, prefill_recent_size = 8):
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric # None, fixed, linear, jump
        self.decoding_window_size = decoding_window_size # b1 + b2
        self.decoding_recent_size = decoding_recent_size # b2
        self.prefill_window_size = prefill_window_size # a1 + a2
        self.prefill_recent_size = prefill_recent_size # a2
        assert self.decoding_window_size - self.decoding_recent_size > 0    
        assert self.prefill_window_size - self.prefill_recent_size > 0
    
    def update_kv(self, key_states, query_states, value_states, num_key_value_groups = None, attention_mask = None):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        # P(value_states)
        # P(key_states)
        if(num_key_value_groups is not None):
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]
        # P(value_states)        
        # reset decoding step       
        ALLKVCluster.current_decoding_step = 0
        ALLKVCluster.jump_step = 0
        
        if(k_len < self.prefill_recent_size + self.prefill_window_size):
            # print(k_len, "-----")
            # P(value_states)
            if(num_key_value_groups is not None):
                key_states = restore_kv(key_states, num_key_value_groups)
                value_states = restore_kv(value_states, num_key_value_groups)
            ALLKVCluster.max_capacity_prompt = key_states.shape[-2]
            return key_states, value_states
        
        a2 = self.prefill_recent_size
        a1 = self.prefill_window_size - self.prefill_recent_size
        # print(a1, a2)
        attention = torch.matmul(query_states, key_states[..., :-a2, :].transpose(2, 3)) / math.sqrt(head_dim)  
        attention = nn.functional.softmax(attention, dim = -1, dtype=torch.float32).to(key_states.dtype)
        attention_sum = attention.sum(dim = -2)
        # print(attention_sum.shape)
        # print(a1)
        a1_indices = attention_sum.topk(a1, dim = -1).indices
        a1_indices = a1_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_a1_compress = key_states[..., :-a2, :].gather(dim = 2, index = a1_indices)
        v_a1_compress = value_states[:, :, :-a2, :].gather(dim = 2, index = a1_indices)
        # print(k_a1_compress)
        # print(v_a1_compress)
        key_states = torch.cat([k_a1_compress, key_states[..., -a2:, :]], dim = 2)
        value_states = torch.cat([v_a1_compress, value_states[..., -a2:, :]], dim = 2)
        ##### Record max_capacity_prompt #####
        ALLKVCluster.max_capacity_prompt = key_states.shape[-2]

        if(num_key_value_groups is not None):
            key_states = restore_kv(key_states, num_key_value_groups)
            value_states = restore_kv(value_states, num_key_value_groups)
        return key_states, value_states
    
    def update_kv_snap(self, key_states, query_states, value_states, max_capacity_prompt = 1024 + 32, window_size = 32, kernel_size = 5, pooling = 'avgpool', num_key_value_groups = None):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -window_size:, -window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim = -2)
            if pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
            elif pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(max_capacity_prompt - window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -window_size:, :]
            v_cur = value_states[:, :, -window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            ALLKVCluster.max_capacity_prompt = key_states.shape[-2]
            return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, num_key_value_groups = None, attention_mask = None):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            # print("ALLKV: no compression")
            return key_states, value_states
        elif self.decoding_metric == 'fixed':
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            
            if k_len < ALLKVCluster.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                # print(attn_weights.shape) # bsz, num_heads, q_len=1, k_len

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                # print(attn_weights.shape)
                
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                # print(attn_weights_sum.shape)
                
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(ALLKVCluster.max_capacity_prompt), dtype=torch.int32).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)
                # print(prefill_indices.shape)

                # decoding cache
                decoding_indices = attn_cache[:, :, ALLKVCluster.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                #top k means the largest k values indices in the decoding cache but we need to add the prefill cache size if we want to get the real indices 
                decoding_indices += ALLKVCluster.max_capacity_prompt
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                # print(decoding_indices.shape)
                
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)
                # print(indices.shape)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'linear':
            raise ValueError("wait implemented") # TODO
        elif self.decoding_metric == 'jump':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + ALLKVCluster.current_decoding_step//(15*32) # TODO: change the step size
            ALLKVCluster.current_decoding_step += 1
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            elif ALLKVCluster.jump_step < 15*32:
                ALLKVCluster.jump_step += 1
                return key_states, value_states
            else:
                # print(f"ALL decoding_window_size {decoding_window_size}")
                ALLKVCluster.jump_step = 0
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                # print(attn_weights.shape) # bsz, num_heads, q_len=1, k_len

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                # print(attn_weights.shape)
                
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                # print(attn_weights_sum.shape)
                
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int32).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)
                # print(prefill_indices.shape)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += self.max_capacity_prompt
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                # print(decoding_indices.shape)
                
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)
                # print(indices.shape)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        else:
            raise ValueError('Decoding metric not supported')
        

    def update_quant_kv_in_decoding(self, query_states, past_key_value, group_size, bits) -> torch.Tensor: 
        
        key_states_quant_trans = past_key_value[0]
        key_states_full = past_key_value[1]
        key_scale_trans = past_key_value[2]
        key_mn_trans = past_key_value[3]
        value_states_quant = past_key_value[4]
        value_states_full = past_key_value[5]
        value_scale = past_key_value[6]
        value_mn = past_key_value[7]
        key_mx_trans = past_key_value[8]
        kv_seq_len = past_key_value[9]
        # print("---", key_states_quant_trans.shape, key_states_full.shape)
        decoding_window_size = self.decoding_window_size # b1 + b2
        window_size = self.decoding_recent_size   # b2
        feat_per_int = 32 // bits
        
        assert group_size % feat_per_int == 0
        
        if key_states_quant_trans is not None:
            bsz, num_heads, head_dim_k, quant_k_len = key_states_quant_trans.shape
        else:
            bsz, num_heads, head_dim_k, quant_k_len = 0, 0, 0, 0
        head_dim_v = value_states_quant.shape[-1] if value_states_quant is not None else 0
        full_v_len = value_states_full.shape[-1] if value_states_full is not None else 0
        full_k_len = key_states_full.shape[-2] if key_states_full is not None else 0
        tot_len = quant_k_len * feat_per_int + full_k_len
        num = group_size // feat_per_int
        if(self.max_capacity_prompt % group_size != 0):
            self.max_capacity_prompt = (self.max_capacity_prompt // group_size + 1) * group_size
        if(self.max_capacity_prompt + decoding_window_size >= tot_len):
            return past_key_value
        # print(f"full_kv_len = {full_k_len}, quant_kv_len = {quant_k_len}, seq_len = {tot_len}, max_capcity_prompt = {self.max_capacity_prompt}, max = {self.max_capacity_prompt + decoding_window_size}")
        
        b1 = decoding_window_size - window_size
        b2_k = window_size - full_k_len 
        b2_v = window_size - full_v_len
        
        
        b1 = (b1 // group_size + 1) * group_size if b1 % group_size != 0 else b1
        b2_k = (b2_k // group_size + 1) * group_size if b2_k % group_size != 0 else b2_k
        b2_v = (b2_v // group_size + 1) * group_size if b2_v % group_size != 0 else b2_v
        if(self.max_capacity_prompt + b1 + b2_k + full_k_len >= tot_len):
            return past_key_value
        if(self.max_capacity_prompt + b1 + b2_v + full_v_len >= tot_len):
            return past_key_value
        # print(b1, b2_k)
        
        assert b2_k >= 0
        # print("need to throw some cache")    
        
        # print(key_states_quant_trans.shape, key_states_full.shape if key_states_full is not None else "None", key_scale_trans.shape, key_mx_trans.shape, key_mn_trans.shape)
        # print(value_states_quant.shape, value_states_full.shape, value_scale.shape, value_mn.shape)
        # prefill cache
        # print(self.max_capacity_prompt, feat_per_int, group_size)
        # prefill_indices_quant_k = torch.tensor(range(self.max_capacity_prompt // feat_per_int), dtype=torch.int32).to(key_states_quant_trans.device)
        prefill_indices_quant_k = torch.arange(self.max_capacity_prompt // feat_per_int, dtype=torch.int32, device=key_states_quant_trans.device)
        # print("______________________")
        prefill_indices_quant_k = prefill_indices_quant_k.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(bsz, num_heads, head_dim_k, 1)
        prefill_indices_quant_v = torch.arange(self.max_capacity_prompt, dtype=torch.int32, device=value_states_quant.device)
        prefill_indices_quant_v = prefill_indices_quant_v.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim_v)        
        prefill_indices_mx_k = torch.arange(self.max_capacity_prompt // group_size, dtype=torch.int32, device=key_mx_trans.device)
        prefill_indices_mx_k = prefill_indices_mx_k.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(bsz, num_heads, head_dim_k, 1)
        prefill_indices_mx_v = torch.arange(self.max_capacity_prompt, dtype=torch.int32, device=value_mn.device)
        prefill_indices_mx_v = prefill_indices_mx_v.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim_v // num)       
        

        
        # b2 % group_size == 0  and self.max_capacity_prompt % group_size == 0 ==>> (tot - b2 - max) % group_size == 0
        key_mx_trans_b1 = key_mx_trans[..., self.max_capacity_prompt // group_size: -b2_k // group_size]
        attention = torch.matmul(query_states, key_mx_trans_b1) / math.sqrt(head_dim_k)
        attention = nn.functional.softmax(attention, dim = -1, dtype=torch.float32).to(key_states_quant_trans.device)
        
        attn_weights_sum = attention.sum(dim = -2)
        
        b1_indices_mx_k = attn_weights_sum.topk(b1 // group_size, dim = -1).indices
        b1_indices_quant_k = get_repeat_tensor(b1_indices_mx_k, num)
        
        #[1, 5, 8] ==>> [1, 2, 10, 11, 16, 17]
        # print(b1, b1_indices_mx_k.shape, b1_indices_quant_k.shape)
        
        b1_indices_quant_k += self.max_capacity_prompt // feat_per_int
        # b1_indices_quant_k.to('cpu')

        b1_indices_quant_v = get_repeat_tensor(b1_indices_mx_k, group_size)
        b1_indices_quant_k = b1_indices_quant_k.unsqueeze(-2).expand(-1, -1, head_dim_k, -1)
        b1_indices_quant_v = b1_indices_quant_v.unsqueeze(-1).expand(-1, -1, -1, head_dim_v)
        # print(b1_indices_quant_k.shape, prefill_indices_quant_k.shape)
        indices_quant_k = torch.cat([prefill_indices_quant_k, b1_indices_quant_k], dim=3)
        indices_quant_v = torch.cat([prefill_indices_quant_v, b1_indices_quant_v], dim=2)
        

        # print(indices_quant_k.shape, indices_quant_v.shape, key_states_quant_trans[..., : -b2_k // feat_per_int].shape)
        key_states_quant_trans_compress = key_states_quant_trans[..., : -b2_k // feat_per_int].gather(dim = 3, index = indices_quant_k)
        # print(key_states_quant_trans_compress.shape)
        # print(b2_v, value_states_quant[..., : -b2_v, :].shape)
        value_states_quant_compress = value_states_quant[..., : -b2_v, :].gather(dim = 2, index = indices_quant_v)
        # print(value_states_quant_compress.shape)
        k_states_quant_trans_cur = key_states_quant_trans[..., -b2_k // feat_per_int:]
        v_states_quant_cur = value_states_quant[..., -b2_v:, :]
        # print(key_states_quant_trans.shape, k_states_quant_trans_cur.shape)
        key_states_quant_trans = torch.cat([key_states_quant_trans_compress, k_states_quant_trans_cur], dim = 3)
        value_states_quant = torch.cat([value_states_quant_compress, v_states_quant_cur], dim=2)
        
        b1_indices_mx_k += self.max_capacity_prompt // group_size
        b1_indices_mx_v = get_repeat_tensor(b1_indices_mx_k, group_size)
        b1_indices_mx_k = b1_indices_mx_k.unsqueeze(-2).expand(-1, -1, head_dim_k, -1)
        b1_indices_mx_v = b1_indices_mx_v.unsqueeze(-1).expand(-1, -1, -1, head_dim_v // num)
        indices_mx_k = torch.cat([prefill_indices_mx_k, b1_indices_mx_k], dim = 3)
        indices_mx_v = torch.cat([prefill_indices_mx_v, b1_indices_mx_v], dim = 2)
        # print(indices_mx_k.shape, indices_mx_v.shape)
        
        key_mx_trans_compress = key_mx_trans[..., : -b2_k // group_size].gather(dim = 3, index = indices_mx_k)
        key_mn_trans_compress = key_mn_trans[..., : -b2_k // group_size].gather(dim = 3, index = indices_mx_k)
        key_scale_trans_compress = key_scale_trans[..., : -b2_k // group_size].gather(dim = 3, index = indices_mx_k)
        # print(head_dim_v, value_mn.shape)
        value_scale_compress = value_scale[..., :-b2_v, :].gather(dim = 2, index = indices_mx_v)
        value_mn_compress = value_mn[..., :-b2_v, :].gather(dim = 2, index = indices_mx_v)
        k_mx_trans_cur = key_mx_trans[..., -b2_k // group_size:]
        k_mn_trans_cur = key_mn_trans[..., -b2_k // group_size:]
        k_scale_trans_cur = key_scale_trans[..., -b2_k // group_size:]
        v_scale_cur = value_scale[..., -b2_v: , :]
        v_mn_cur = value_mn[..., -b2_v: , :]
        key_mx_trans = torch.cat([key_mx_trans_compress, k_mx_trans_cur], dim=3)
        key_mn_trans = torch.cat([key_mn_trans_compress, k_mn_trans_cur], dim=3)
        key_scale_trans = torch.cat([key_scale_trans_compress, k_scale_trans_cur], dim=3)
        value_scale = torch.cat([value_scale_compress, v_scale_cur], dim=2)
        value_mn = torch.cat([value_mn_compress, v_mn_cur], dim=2)
        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, key_mx_trans, kv_seq_len)
        # print(key_states_quant_trans.shape, key_states_full.shape if key_states_full is not None else "None", key_scale_trans.shape, key_mx_trans.shape, key_mn_trans.shape)
        # print(value_states_quant.shape, value_states_full.shape, value_scale.shape, value_mn.shape)
        # print("##############################################################################")
        return past_key_value
        
        

    
def init_ALLKV(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 512
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 128
        if not hasattr(self.config, 'prefill_windown_size'):
            self.config.prefill_window_size = 2048
        if not hasattr(self.config, 'prefill_recent_size'):
            self.config.prefill_recent_size = 8  # 8
    
    
    self.kv_cluster = ALLKVCluster(
        decoding_metric=self.config.decoding_metric,
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        prefill_window_size=self.config.prefill_window_size,
        prefill_recent_size=self.config.prefill_recent_size
        )
