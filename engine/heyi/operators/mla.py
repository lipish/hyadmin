"""
Description  :
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
"""

import logging
from typing import Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache

from heyi.models.deepseek_v3 import DeepseekV3Config, DeepseekV3Attention, apply_rotary_pos_emb

try:
    from flash_attn import flash_attn_func
except:
    pass

from heyi.operators.triton_attention import decode_attention_fwd_grouped
from heyi.operators.triton_attention_prefill import context_attention_fwd
import flashinfer

logger = logging.getLogger("attention")


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class MLA(DeepseekV3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    attn_mask: Optional[torch.Tensor] = None

    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: Optional[int] = None,
        absorb_for_prefill: bool = False,
    ):
        super().__init__(config, layer_idx)
        self.attn_wrapper: Optional[flashinfer.BatchMLAPagedAttentionWrapper] = None
        self.absorb_for_prefill = absorb_for_prefill

    def get_absorbed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (hasattr(self, "q_absorb") and hasattr(self, "out_absorb")):
            kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
            self.q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :].view(
                self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank
            )
            self.out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :].view(
                self.num_heads, self.v_head_dim, self.kv_lora_rank
            )

        return self.q_absorb, self.out_absorb

    def forward_linux_triton(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        compressed_kv = compressed_kv.view(bsz, q_len, 1, self.kv_lora_rank)

        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since transformer version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, unsqueeze_dim=2)
        # q_pe [bsz, q_len, self.num_heads, self.qk_rope_head_dim] k_pe [bsz, q_len, 1, self.qk_rope_head_dim]
        
        # decode
        if q_len == 1:
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                compressed_kv_with_k_pe, page_table = past_key_value.update(compressed_kv, k_pe, self.layer_idx, cache_kwargs)
                compressed_kv = compressed_kv_with_k_pe [:, :, :, :self.kv_lora_rank] # for speed
                # compressed_kv_with_k_pe [bsz, q_len, 1, self.kv_lora_rank + self.qk_rope_head_dim]
                # compressed_kv [bsz, q_len, 1, self.kv_lora_rank]

            # q_nope [bsz, q_len, self.num_heads, self.qk_nope_head_dim]
            # q_absorb [self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank]
            q_absorb, out_absorb = self.get_absorbed()
            q_nope = q_nope.transpose(1, 2) # q_len is 1, no GPU overhead, same below
            q_nope = torch.matmul(q_nope, q_absorb) # batched MM
            q_nope = q_nope.transpose(1, 2)
            #assert q_nope.is_contiguous()
            
            # q_nope [bsz, q_len, self.num_heads, self.kv_lora_rank]
            # q_pe [bsz, q_len, self.num_heads, self.qk_rope_head_dim]
            query_states = torch.cat([q_nope, q_pe], dim=-1)
            
            query_states = query_states.squeeze(1)
            attn_output = torch.zeros_like(q_nope) # [bsz, q_len, self.num_heads, self.kv_lora_rank]
            
            attn_logits = torch.empty(
                    (
                        bsz,
                        self.num_heads,
                        4, #num_kv_splits # follow vLLM, fix it TODO
                        self.kv_lora_rank + 1, 
                    ),
                    dtype=torch.float32,
                    device = attn_output.device
                )

            """
            print("query_states", torch.isnan(query_states).any())
            print("compressed_kv_with_k_pe", torch.isnan(compressed_kv_with_k_pe[:,:,0,:]).any())
            print("compressed_kv", torch.isnan(compressed_kv[:,:,0,:]).any())
            print("position_ids", torch.isnan(position_ids).any())
            """

            # flash attn doesn't support head_dim bigger than 256
            # use triton attention kernel adapted from vLLM and SGLang for MQA
            decode_attention_fwd_grouped(query_states, compressed_kv_with_k_pe, compressed_kv, attn_output,
                             page_table,
                             position_ids.squeeze(0).to(torch.int32)+1, attn_logits,
                             4, #num_kv_splits # follow vLLM, fix it TODO
                             self.softmax_scale,
                             past_key_value.page_size)
            
            # attn_output [bsz, q_len, self.num_heads, self.kv_lora_rank]
            # out_absorb [self.num_heads, self.v_head_dim, self.kv_lora_rank]
            attn_output = attn_output.transpose(1, 2)
            attn_output = torch.matmul(attn_output, out_absorb.mT)
            attn_output = attn_output.transpose(1, 2)
            
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
            attn_output = self.o_proj(attn_output)
            
            #print("attn_output", torch.isnan(attn_output).any())
            return attn_output, None, past_key_value
        else:
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                k_pe.squeeze(0)
                compressed_kv.squeeze(0)
                compressed_kv_with_k_pe, _ = past_key_value.update(compressed_kv, k_pe, self.layer_idx, cache_kwargs)
                compressed_kv, k_pe = torch.split(
                    compressed_kv_with_k_pe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            k_pe = k_pe.view(bsz, -1, self.qk_rope_head_dim)
            k_pe = k_pe[:, :kv_seq_len]
            compressed_kv = compressed_kv.view(bsz, -1, self.kv_lora_rank)
            compressed_kv = compressed_kv[:, :kv_seq_len]
            kv = (
                self.kv_b_proj(compressed_kv)
                .view(bsz, kv_seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            )
            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            query_states = k_pe.new_empty(bsz, q_len, self.num_heads, self.q_head_dim)
            query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
            query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

            key_states = k_pe.new_empty(bsz, kv_seq_len, self.num_heads, self.q_head_dim)
            key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
            key_states[:, :, :, self.qk_nope_head_dim:] = k_pe.view(bsz, kv_seq_len, 1, -1)
            
            value_states = value_states.view(bsz, kv_seq_len, self.num_heads, self.v_head_dim)
            value_states_padded = torch.nn.functional.pad(value_states, [0, query_states.shape[-1] - value_states.shape[-1]], value=0)

            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states_padded,
                softmax_scale=self.softmax_scale,
                causal=True,
            )

            if self.q_head_dim != self.v_head_dim:
                attn_output = attn_output[:, :, :, : self.v_head_dim]

            attn_output = attn_output.reshape(
                bsz, q_len, self.num_heads * self.v_head_dim
            ).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value


    def forward_linux_flashinfer(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        compressed_kv = compressed_kv.view(bsz, q_len, 1, self.kv_lora_rank)

        kv_seq_len = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    "The cache structure has changed since version transformer verision v4.36. If you are using"
                    f" {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to"
                    " initialize the attention class with a layer index."
                )
            kv_seq_len = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(q_pe, position_ids)
        if bsz > 1:
            cos = cos.transpose(1, 0)
            sin = sin.transpose(1, 0)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, None, unsqueeze_dim=2)
        # q_pe [bsz, q_len, self.num_heads, self.qk_rope_head_dim] k_pe [bsz, q_len, 1, self.qk_rope_head_dim]

        # decode
        if q_len == 1 or self.absorb_for_prefill:
            if past_key_value is not None:
                compressed_kv_with_k_pe = past_key_value.update(
                    compressed_kv, k_pe, self.layer_idx,
                )
                compressed_kv = compressed_kv_with_k_pe[:, :, : self.kv_lora_rank].view(
                    -1, past_key_value.page_size, self.kv_lora_rank
                )
                k_pe = compressed_kv_with_k_pe[:, :, self.kv_lora_rank :].view(
                    -1, past_key_value.page_size, self.qk_rope_head_dim
                )
                # k_pe [max_pages, page_size, self.qk_rope_head_dim]
                # compressed_kv [max_pages, page_size, self.kv_lora_rank]

            # q_nope [bsz, q_len, self.num_heads, self.qk_nope_head_dim]
            # q_absorb [self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank]
            q_absorb, out_absorb = self.get_absorbed()
            q_nope = q_nope.transpose(1, 2)  # q_len is 1, no GPU overhead, same below
            q_nope = torch.matmul(q_nope, q_absorb)  # batched MM
            q_nope = q_nope.transpose(1, 2)
            q_nope = q_nope.contiguous()
            # assert q_nope.is_contiguous()

            # q_nope [bsz, q_len, self.num_heads, self.kv_lora_rank]
            # q_pe [bsz, q_len, self.num_heads, self.qk_rope_head_dim]
            if bsz > 1:
                q_nope.squeeze_(1)
                q_pe.squeeze_(1)
            else:
                q_nope.squeeze_(0)
                q_pe.squeeze_(0)

            # flash attn doesn't support head_dim bigger than 256, use flashinfer
            attn_wrapper = kwargs.get("attn_wrapper")

            attn_output = attn_wrapper.run(q_nope, q_pe, compressed_kv, k_pe).view(
                bsz, q_len, self.num_heads, self.kv_lora_rank
            )

            # attn_wrapper run output: [tokens, self.num_heads, self.kv_lora_rank]
            # attn_output [bsz, q_len, self.num_heads, self.kv_lora_rank]
            # out_absorb [self.num_heads, self.v_head_dim, self.kv_lora_rank]
            attn_output = attn_output.transpose(1, 2)  # [bsz, self.num_heads, q_len, self.kv_lora_rank]
            attn_output = torch.matmul(attn_output, out_absorb.mT)  # [bsz, self.num_heads, q_len, self.v_head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()  # [bsz, q_len, self.num_heads, self.kv_lora_rank]

            attn_output = attn_output.reshape(
                bsz, q_len, self.num_heads * self.v_head_dim
            )  # [bsz, q_len, self.num_heads * self.v_head_dim]
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value
        else:
            if past_key_value is not None:
                k_pe.squeeze(0)
                compressed_kv.squeeze(0)
                compressed_kv_with_k_pe = past_key_value.update(compressed_kv, k_pe, self.layer_idx)
                compressed_kv_with_k_pe = compressed_kv_with_k_pe[past_key_value.buffers["page_indices"]]
                compressed_kv, k_pe = torch.split(
                    compressed_kv_with_k_pe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            k_pe = k_pe.view(bsz, -1, self.qk_rope_head_dim)
            k_pe = k_pe[:, :kv_seq_len]
            compressed_kv = compressed_kv.view(bsz, -1, self.kv_lora_rank)
            compressed_kv = compressed_kv[:, :kv_seq_len]
            kv = self.kv_b_proj(compressed_kv).view(
                bsz, kv_seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            query_states = k_pe.new_empty(bsz, q_len, self.num_heads, self.q_head_dim)
            query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
            query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

            key_states = k_pe.new_empty(bsz, kv_seq_len, self.num_heads, self.q_head_dim)
            key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
            key_states[:, :, :, self.qk_nope_head_dim :] = k_pe.view(bsz, kv_seq_len, 1, -1)

            value_states = value_states.view(bsz, kv_seq_len, self.num_heads, self.v_head_dim)
            value_states_padded = torch.nn.functional.pad(
                value_states, [0, query_states.shape[-1] - value_states.shape[-1]], value=0
            )

            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states_padded,
                softmax_scale=self.softmax_scale,
                causal=True,
            )

            if self.q_head_dim != self.v_head_dim:
                attn_output = attn_output[:, :, :, : self.v_head_dim]

            attn_output = attn_output.reshape(
                bsz, q_len, self.num_heads * self.v_head_dim
            ).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        return self.forward_linux_flashinfer(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            **kwargs,
        )
        # else:
        #     return self.forward_linux_triton(
        #         hidden_states,
        #         attention_mask,
        #         position_ids,
        #         past_key_value,
        #         output_attentions,
        #         use_cache,
        #         cache_position,
        #         **kwargs,
        #     )

