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


def get_repeat_tensor(x: torch.Tensor, g: int):
    """
    [x1, x2, x3, ... , xn] to 
    [x1 * g, x1 * g + 1, x1 * g + 2, ..., x1 * g + g - 1,
     x2 * g, x2 * g + 1, x2 * g + 2, ..., x2 * g + g - 1,
     ...
    xn * g, xn * g + 1, xn * g + 2, ..., xn * g + g - 1]
    """
    indices = torch.arange(g)
    expanded_indices = indices.repeat(x.shape[-1], 1)
    # print('expanded_indices', expanded_indices)
    # 计算结果
    result = x.unsqueeze(-1) * g + expanded_indices
    # print('result', result)
    # 展平结果张量
    res = torch.flatten(result, start_dim=-2)
    
    return res


class PyramidKVCluster():
    def __init__(self, decoding_metric = 'None', num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool', beta = 20, num_layers = 80, layer_idx=None):
        
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        self.steps = -1
        self.beta = beta
        
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0

    def reset(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num
        
            
        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt - self.window_size) * 2 - max_num
    
       
        steps = (max_num - min_num) // self.num_hidden_layers
        max_capacity_prompt = max_num - self.layer_idx * steps
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            # attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim) # Pyramidkv(snapkv)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim) # Pyramidinfer(h2o)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2) # Pyramidkv(snapkv)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2) # Pyramidinfer(h2o)
            ## PyramidKV(snapkv)
            # if self.pooling == 'avgpool':
            #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # elif self.pooling == 'maxpool':
            #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # else:
            #     raise ValueError('Pooling method not supported')
            attn_cache = attn_weights_sum # PyrmamidInfer(h2o)
            indices = attn_cache.topk(self.max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        else:
            # attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim) # Pyramidkv(snapkv)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim) # Pyramidinfer(h2o)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2) # Pyramidkv(snapkv)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2) # Pyramidinfer(h2o)
            ## PyramidKV(snapkv)
            # if self.pooling == 'avgpool':
            #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # elif self.pooling == 'maxpool':
            #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # else:
            #     raise ValueError('Pooling method not supported')
            attn_cache = attn_weights_sum # PyrmamidInfer(h2o)
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        elif self.decoding_metric == 'pyramidinfer':
            # prefill+decoding cache, compute the number of tokens to keep in the cache
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            min_num = (self.max_capacity_prompt + decoding_window_size - window_size) // 2 # TODO beta set to 2
            max_num = (self.max_capacity_prompt + decoding_window_size - window_size) * 2 - min_num

            steps = (max_num - min_num) // self.num_hidden_layers
            max_capacity_prompt = max_num - self.layer_idx * steps

            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            elif k_len < (self.max_capacity_prompt - window_size) * 2 + decoding_window_size:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                # decoding cache
                decoding_indices = attn_cache.topk(self.max_capacity_prompt + decoding_window_size - window_size, dim=-1).indices
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices = decoding_indices
                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                # decoding cache
                decoding_indices = attn_cache.topk(max_capacity_prompt + decoding_window_size, dim=-1).indices
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices = decoding_indices
                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        else:
            raise ValueError('Decoding metric not supported')


class SnapKVCluster():
    def __init__(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0

    def reset(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        else:
            # TODO
            raise ValueError('Decoding metric not supported')


class H2OKVCluster():

    current_decoding_step = 0
    jump_step = 0
    jump_layer = 0

    def __init__(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add H2O delta #####
        self.delta = delta
        ##### Add H2O num_hidden_layers #####
        self.num_hidden_layers = num_hidden_layers

    def reset(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add H2O delta #####
        self.delta = delta
        ##### Add H2O num_hidden_layers #####
        self.num_hidden_layers = num_hidden_layers

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # reset decoding step
        H2OKVCluster.current_decoding_step = 0
        H2OKVCluster.jump_step = 0
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, :, : -self.window_size].sum(dim = -2)
            attn_cache = attn_weights_sum
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        elif self.decoding_metric == 'h2o':
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                # decoding cache
                decoding_indices = attn_cache.topk(self.max_capacity_prompt + decoding_window_size - window_size, dim=-1).indices
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                indices = decoding_indices
                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'fixed':
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim) # bsz, num_heads, q_len=1, k_len
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'linear':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + H2OKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            H2OKVCluster.current_decoding_step += 1
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'jump':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + H2OKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            H2OKVCluster.current_decoding_step += 1
            
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            elif H2OKVCluster.jump_step < self.delta*self.num_hidden_layers:
                H2OKVCluster.jump_step += 1
                return key_states, value_states
            else:
                H2OKVCluster.jump_layer += 1
                if H2OKVCluster.jump_layer == self.num_hidden_layers:
                    H2OKVCluster.jump_step = 0
                    H2OKVCluster.jump_layer = 0
                
                
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(self.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, self.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        else:
            raise ValueError('Decoding metric not supported')


class StreamingLLMKVCluster():
    def __init__(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0

    def reset(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:    
            indices = torch.tensor(range(self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states
        
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        elif self.decoding_metric == 'slm':
            if k_len < self.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                decoding_window_size = self.decoding_window_size
                window_size = self.decoding_recent_size
                
                # decoding cache
                decoding_indices = torch.tensor(range(self.max_capacity_prompt+decoding_window_size-window_size), dtype=torch.int64).to(key_states.device)
                decoding_indices = decoding_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)
                # print(decoding_indices.shape)
                
                indices = decoding_indices
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

        
class ALLKVCluster():
    
    allkv_max_capacity_prompt = 0
    current_decoding_step = 0
    jump_step = 0

    def __init__(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, prefill_window_size = 2048, prefill_recent_size = 32):
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric # None, fixed, linear, jump
        self.decoding_window_size = decoding_window_size # b1 + b2
        self.decoding_recent_size = decoding_recent_size # b2
        self.prefill_window_size = prefill_window_size # a1 + a2
        self.prefill_recent_size = prefill_recent_size # a2
        assert self.decoding_window_size - self.decoding_recent_size > 0
        assert self.prefill_recent_size - self.prefill_recent_size > 0            

    def reset(self, decoding_metric = 'None', decoding_window_size = 1024, decoding_recent_size = 256, prefill_window_size = 2048, prefill_recent_size = 32):
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric # None, fixed, linear, jump
        self.decoding_window_size = decoding_window_size # b1 + b2
        self.decoding_recent_size = decoding_recent_size # b2
        self.prefill_window_size = prefill_window_size # a1 + a2
        self.prefill_recent_size = prefill_recent_size # a2
        assert self.decoding_window_size - self.decoding_recent_size > 0    
        assert self.prefill_recent_size - self.prefill_recent_size > 0
    
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]
        
        # reset decoding step       
        ALLKVCluster.current_decoding_step = 0
        ALLKVCluster.jump_step = 0
        
        if(k_len < self.prefill_recent_size + self.prefill_window_size):
            ALLKVCluster.max_capacity_prompt = key_states.shape[-2]
            return key_states, value_states
        
        a2 = self.prefill_recent_size
        a1 = self.prefill_window_size - self.prefill_recent_size
        
        attention = torch.matmul(query_states, key_states[..., :-a2, :].transpose(2, 3)) / math.sqrt(head_dim)  
        attention = nn.functional.softmax(attention, dim = -1, dtype=torch.float32).to(key_states.dtype)
        attention_sum = attention.sum(dim = -2)
        
        a1_indices = attention_sum.topk(a1, dim = -2).indices
        a1_indices = a1_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_a1_compress = key_states[..., :-a2, :].gather(dim = 2, index = a1_indices)
        v_a1_compress = value_states[:, :, :-a2, :].gather(dim = 2, index = a1_indices)
        
        key_states = torch.cat([k_a1_compress, key_states[..., -a2:, :]], dim = 2)
        value_states = torch.cat([v_a1_compress, value_states[..., -a2:, :]], dim = 2)
        ##### Record max_capacity_prompt #####
        ALLKVCluster.max_capacity_prompt = key_states.shape[-2]

        return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
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
        

    def update_quant_kv_in_decoding(self, query_states, past_key_value, attention_mask, num_key_value_groups, group_size, bits)->torch.Tensor: 
        
        key_states_quant_trans = past_key_value[0]
        key_states_full = past_key_value[1]
        key_scale_trans = past_key_value[2]
        key_mn_trans = past_key_value[3]
        value_states_quant = past_key_value[4]
        value_states_full = past_key_value[5]
        value_scale = past_key_value[6]
        value_mn = past_key_value[7]
        key_mx_trans = past_key_value[8]
                
        
        decoding_window_size = self.decoding_window_size
        window_size = self.decoding_recent_size       
        feat_per_int = 32 / bits
        
        assert group_size % feat_per_int == 0
        
        bsz, num_heads, head_dim, quant_k_len = key_states_quant_trans.shape
        full_k_len = key_states_full.shape[-1]
        
        if(self.max_capacity_prompt % group_size != 0):
            self.max_capacity_prompt = (self.max_capacity_prompt // group_size + 1) * group_size
        
        if(self.max_capacity_prompt + decoding_window_size > quant_k_len * feat_per_int + full_k_len):
            return past_key_value
        
        b1 = decoding_window_size - window_size
        b2 = window_size - full_k_len
        
        assert b2 >= 0
        
        # prefill cache
        prefill_indices_quant = torch.tensor(range(self.max_capacity_prompt // feat_per_int), dtype=torch.int32).to(key_states_quant_trans.device)
        prefill_indices_quant = prefill_indices_quant.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(bsz, num_heads, head_dim, 1)
        prefill_indices_mx = torch.tensor(range(self.max_capacity_prompt // group_size), dtype=torch.int32).to(key_states_quant_trans.device)
        prefill_indices_mx = prefill_indices_mx.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(bsz, num_heads, head_dim, 1)
        
        
        if(b2 % group_size != 0):
            b2 = (b2 // group_size + 1) * group_size
        if(b1 % group_size != 0):
            b1 = (b1 // group_size + 1) * group_size
        
        # b2 % group_size == 0  and self.max_capacity_prompt % group_size == 0 ==>> (tot - b2 - max) % group_size == 0
        key_mx_trans_b1 = key_mx_trans[..., self.max_capacity_prompt // group_size: -b2 // group_size]
        attention = torch.matmul(query_states, key_mx_trans_b1) / math.sqrt(head_dim)
        attention = nn.functional.softmax(attention, dim = -1, dtype=torch.float32).to(key_states_quant_trans.device)
        
        attn_weights_sum = attention.sum(dim = -2)
        
        b1_indices_mx = attn_weights_sum.topk(b1 // group_size, dim = -1).indices
        b1_indices_quant = get_repeat_tensor(b1_indices_mx, group_size // feat_per_int)
        
        
        
        
        #[1, 5, 8] ==>> [1, 2, 10, 11, 16, 17]
        b1_indices_quant += self.max_capacity_prompt // feat_per_int
        b1_indices_quant = b1_indices_quant.unsqueeze(-2).expand(-1, -1, head_dim, -1)

        indices_quant_k = torch.cat([prefill_indices_quant, b1_indices_quant], dim=2)
        indices_quant_v = indices_quant_k.transpose(2, 3)
        
        key_states_quant_trans_compress = key_states_quant_trans[..., : -b2 // feat_per_int].gather(dim = 3, index = indices_quant_k)
        value_states_quant_compress = value_states_quant[..., : -b2 // feat_per_int].gather(dim = 2, index = indices_quant_v)
        k_states_quant_trans_cur = key_states_quant_trans[..., -b2 // feat_per_int]
        v_states_quant_cur = value_states_quant[..., -b2 // feat_per_int, :]
        key_states_quant_trans = torch.cat([key_states_quant_trans_compress, k_states_quant_trans_cur], dim = 3)
        value_states_quant = torch.cat([value_states_quant_compress, v_states_quant_cur], dim=2)
        
        b1_indices_mx += self.max_capacity_prompt // group_size
        b1_indices_mx = b1_indices_mx.unsqueeze(-2).expand(-1, -1, head_dim, -1)
        
        indices_mx_k = torch.cat([prefill_indices_mx, b1_indices_mx], dim=2)
        indices_mx_v = indices_mx_k.transpose(2, 3)
        
        key_mx_trans_compress = key_mx_trans[..., : -b2 // group_size].gather(dim = 3, index = indices_mx_k)
        key_mn_trans_compress = key_mn_trans[..., : -b2 // group_size].gather(dim = 3, index = indices_mx_k)
        key_scale_trans_compress = key_scale_trans[..., : -b2 // group_size].gather(dim = 3, index = indices_mx_k)
        value_scale_compress = value_scale[..., : -b2 // group_size].gather(dim = 2, index = indices_mx_v)
        value_mn_compress = value_mn[..., : -b2 // group_size].gather(dim = 2, index = indices_mx_v)
        k_mx_trans_cur = key_mx_trans[..., -b2 // group_size]
        k_mn_trans_cur = key_mn_trans[..., -b2 // group_size]
        k_scale_trans_cur = key_scale_trans[..., -b2 // group_size]
        v_scale_cur = value_scale[..., -b2 // group_size, :]
        v_mn_cur = value_mn[..., -b2 // group_size, :]
        key_mx_trans = torch.cat([key_mx_trans_compress, k_mx_trans_cur], dim=3)
        key_mn_trans = torch.cat([key_mn_trans_compress, k_mn_trans_cur], dim=3)
        key_scale_trans = torch.cat([key_scale_trans_compress, k_scale_trans_cur], dim=3)
        value_scale = torch.cat([value_scale_compress, v_scale_cur], dim=2)
        value_mn = torch.cat([value_mn_compress, v_mn_cur], dim=2)
        
        past_key_value[0] = key_states_quant_trans
        past_key_value[1] = key_states_full
        past_key_value[2] = key_scale_trans
        past_key_value[3] = key_mn_trans
        past_key_value[4] = value_states_quant
        past_key_value[5] = value_states_full
        past_key_value[6] = value_scale
        past_key_value[7] = value_mn
        past_key_value[8] = key_mx_trans
        
        return past_key_value
        
        
def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
    
    
    self.kv_cluster = PyramidKVCluster( 
        num_hidden_layers = num_hidden_layers,
        layer_idx = self.layer_idx,
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        decoding_metric=self.config.decoding_metric
        )
 
def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
    
    self.kv_cluster = SnapKVCluster( 
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        decoding_metric=self.config.decoding_metric
        )

def init_H2O(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
        if not hasattr(self.config, 'delta'):
            self.config.delta = 15
    
    
    self.kv_cluster = H2OKVCluster(
        num_hidden_layers = num_hidden_layers,
        delta=self.config.delta,
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        decoding_metric=self.config.decoding_metric
        )

def init_StreamingLLM(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
    
    
    self.kv_cluster = StreamingLLMKVCluster(
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        decoding_metric=self.config.decoding_metric
        )
    
def init_ALLKV(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'decoding_window_size'):
            self.config.decoding_window_size = 1024
        if not hasattr(self.config, 'decoding_recent_size'):
            self.config.decoding_recent_size = 256
        if not hasattr(self.config, 'prefill_windown_size'):
            self.config.prefill_window_size = 2048
        if not hasattr(self.config, 'prefill_recent_size'):
            self.config.prefill_recent_size = 32  # 8
    
    
    self.kv_cluster = ALLKVCluster(
        decoding_metric=self.config.decoding_metric,
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        decoding_prefill_size=self.config.prefill_window_size,
        decoding_prefill_recent_size=self.config.prefill_recent_size
        )
