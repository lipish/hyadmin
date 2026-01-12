import logging
from typing import Optional, Tuple, Union

import flashinfer
import torch
from transformers.cache_utils import Cache

from heyi.models.qwen3_moe import Qwen3MoeAttention, Qwen3MoeConfig

logger = logging.getLogger("attention")

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class GQA(Qwen3MoeAttention):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int = 0,
    ):
        super().__init__(config, layer_idx)
        self.attn_wrapper = None
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

    # Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        B, q_len = hidden_states.shape[:-1]
        hidden_shape = (B, q_len, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states = query_states.view(B, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(B, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(B, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        if B > 1:
            cos = cos.transpose(1, 0)
            sin = sin.transpose(1, 0)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(
            B, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            B, q_len, self.num_key_value_heads, self.head_dim
        )

        assert past_key_value is not None
        page_table = past_key_value.update(
            key_states, value_states, self.layer_idx
        )

        attn_wrapper: Union[
            flashinfer.BatchDecodeWithPagedKVCacheWrapper,
            flashinfer.BatchPrefillWithPagedKVCacheWrapper,
        ] = kwargs.get("attn_wrapper")

        attn_output = attn_wrapper.run(query_states, page_table)

        attn_output = self.o_proj(attn_output.view(B, q_len, self.num_heads * self.head_dim))

        return attn_output, None
