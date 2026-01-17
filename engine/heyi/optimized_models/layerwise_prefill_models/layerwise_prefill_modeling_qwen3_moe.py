from transformers.processing_utils import Unpack
import re
import time
from typing import List, Optional, Tuple, Union

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
import torch
from torch import nn
from torch.nn import functional as F

from heyi.models.qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeDecoderLayer,
    Qwen3MoeConfig,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)

from heyi.operators import GQA, FusedRMSNorm

from heyi.operators.linear import KLinearTorch
from heyi.utils import main_model_registry
from heyi.utils.stream_loader import StreamLoader, register_opevs
from .common import LP_LinearFP8_FromGPU, LP_LinearTorch_FromGPU, LP_MLP_FromCPUWithBuffer, LP_MLP_FromGPU
from heyi.utils.ring_buffer import RingBufferMgr

from heyi.utils.kvcache.kvcache import PagedGQACache
from heyi.config import Config
from heyi.utils.log import logger
import flashinfer


class LPQwen3MoeAttention(GQA):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        Linear = LP_LinearFP8_FromGPU if config.weight_dtype == torch.float8_e4m3fn else LP_LinearTorch_FromGPU
        self.q_proj = Linear(self.q_proj)
        self.k_proj = Linear(self.k_proj)
        self.v_proj = Linear(self.v_proj)
        self.o_proj = Linear(self.o_proj)

        self.q_norm = FusedRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = FusedRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: PagedGQACache = None,
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

        kv_seq_len = past_key_value.get_usable_length(q_len, self.layer_idx)

        assert past_key_value is not None
        with torch.cuda.device(0):
            kv_states = past_key_value.update(
                key_states.to(0), value_states.to(0), self.layer_idx
            )
        kv_states = kv_states[past_key_value.buffers["page_indices"]].to(Config().layerwise_prefill_device)
        # npages, k&v, page_size, nheads, head_dim

        kv_states = kv_states.transpose(1, 0).reshape(2, -1, self.num_key_value_heads, self.head_dim)[:, :kv_seq_len]

        key_states = kv_states[0].view(-1, self.num_key_value_heads, self.head_dim)
        value_states = kv_states[1].view(-1, self.num_key_value_heads, self.head_dim)
        del kv_states

        attn_wrapper: flashinfer.BatchPrefillWithRaggedKVCacheWrapper = kwargs.get("attn_wrapper")

        attn_output = attn_wrapper.run(query_states, key_states, value_states)
        del query_states, key_states, value_states

        attn_output = self.o_proj(attn_output.view(B, q_len, self.num_heads * self.head_dim))

        return attn_output, None


    def on(self):
        self.q_proj.on()
        self.k_proj.on()
        self.v_proj.on()
        self.o_proj.on()

    def off(self):
        self.q_proj.off()
        self.k_proj.off()
        self.v_proj.off()
        self.o_proj.off()


class LPQwen3MoeSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    ring_buffer_mgr = RingBufferMgr()

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.experts = nn.ModuleList(
            [
                LP_MLP_FromCPUWithBuffer(
                    config,
                    intermediate_size=config.moe_intermediate_size,
                    ring_buffer_mgr=LPQwen3MoeSparseMoeBlock.ring_buffer_mgr,
                )
                for i in range(config.num_experts)
            ]
        )
        # self.gate = KMoEGate(config)
        self.experts_opevs = register_opevs([expert for expert in self.experts])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):

            # print(f"[MODEL] WAIT FOR EXP {expert.key} EV_ON")
            # torch.cuda.nvtx.range_push(f"[MODEL] WAIT FOR {expert.key} EV_ON")
            self.experts_opevs[expert_idx].ev.wait_on()
            # torch.cuda.nvtx.range_pop()
            # print(f"[MODEL] CONFIRMED {expert.key} EV_ON")

            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() == 0:
                # print(f"[MODEL] SKIP EXP {expert.key} EV_OFF")
                self.experts_opevs[expert_idx].ev.set_off()
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            torch.cuda.current_stream().synchronize()

            # print(f"[MODEL] SET EXP {expert.key} EV_OFF")
            self.experts_opevs[expert_idx].ev.set_off()

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states
    


class LPQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = LPQwen3MoeAttention(config, layer_idx)
        attn_opev = register_opevs([self.self_attn])

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = LPQwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = LP_MLP_FromGPU(config, intermediate_size=config.intermediate_size)
        mlp_opev = register_opevs([self.mlp]) if isinstance(self.mlp, LP_MLP_FromGPU) else []
        self.stream_load_opevs = attn_opev + mlp_opev

        self.input_layernorm = FusedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = FusedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        device = torch.cuda.current_device()

        # residual = hidden_states
        if hidden_states.shape[1] > 50000:
            print(f"CONTEXT_LEN={hidden_states.shape[1]}, ENABLE RESIDUAL OFFLOADING")
            offload_residuals = True
        else:
            offload_residuals = False

        residual = hidden_states.clone()
        if offload_residuals:
            residual = residual.to("cpu", non_blocking=True).pin_memory()

        hidden_states.copy_(self.input_layernorm(hidden_states))

        # print(f"[MODEL]: WAIT FOR ATTN {self.self_attn.key} EV_ON")
        self.stream_load_opevs[0].ev.wait_on()
        # print(f"[MODEL]: CONFIRMED {self.self_attn.key} EV_ON")

        # Self Attention
        hidden_states[:], self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        # print(f"[MODEL]: SET {self.self_attn.key} EV_OFF")
        self.stream_load_opevs[0].ev.set_off()

        # hidden_states = residual + hidden_states
        if offload_residuals:
            residual = residual.to(device, non_blocking=True)
        hidden_states.add_(residual)

        # mlp
        # residual = hidden_states
        residual.copy_(hidden_states)

        if offload_residuals:
            residual = residual.to("cpu", non_blocking=True)
        hidden_states.copy_(self.post_attention_layernorm(hidden_states))

        if isinstance(self.mlp, Qwen3MoeSparseMoeBlock):
            pass
        else:
            # print(f"[MODEL]: WAIT FOR MLP {self.mlp.key} EV_ON")
            self.stream_load_opevs[1].ev.wait_on()
            # print(f"[MODEL]: CONFIRMED {self.mlp.key} EV_ON")

        hidden_states.copy_(self.mlp(hidden_states))
        torch.cuda.synchronize()

        if isinstance(self.mlp, Qwen3MoeSparseMoeBlock):
            pass
        else:
            # print(f"[MODEL]: SET {self.mlp.key} EV_OFF")
            self.stream_load_opevs[1].ev.set_off()

        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        if offload_residuals:
            residual = residual.to(device, non_blocking=True)
        # hidden_states = residual + hidden_states
        hidden_states.add_(residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs
    


class LPQwen3MoeModel(Qwen3MoeModel):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)

        self.layers = nn.ModuleList([
            LPQwen3MoeDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
            # for layer_idx in range(1)
        ])
        # self.norm = FusedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        with torch.device("cuda"):
            self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: PagedGQACache = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attn_wrapper = None,
        **flash_attn_kwargs,
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            cuda_device = torch.device('cuda', torch.cuda.current_device())
            self.embed_tokens = self.embed_tokens.to(cuda_device)
            inputs_embeds = self.embed_tokens(input_ids).bfloat16()
            self.embed_tokens = self.embed_tokens.to("cpu")

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )
        causal_mask = None

        hidden_states = inputs_embeds
        del inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        import time
        base = time.perf_counter()
        last_layer_timestamp = time.perf_counter()
        for i, decoder_layer in enumerate(self.layers):
            # torch.cuda.nvtx.range_push("[MODEL] ONLOAD KVCache")
            if Config().layerwise_prefill_device == 0:
                past_key_values.to_(0, i)
            # torch.cuda.nvtx.range_pop()

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                attn_wrapper=attn_wrapper,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

            # torch.cuda.nvtx.range_push("[MODEL] OFFLOAD KVCache")
            if Config().layerwise_prefill_device == 0:
                past_key_values.to_("cpu", i)
            # torch.cuda.nvtx.range_pop()
            print(f"[MODEL] layer {i}: {time.perf_counter() - last_layer_timestamp:.03f}s")
            last_layer_timestamp = time.perf_counter()
            
        hidden_states = self.norm(hidden_states)
        logger.info(f"[MODEL] all layers: {time.perf_counter() - base:.03f}s")

        logger.debug(f"{hidden_states=}")

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
    


class LPQwen3MoeForCausalLM(Qwen3MoeForCausalLM):

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.config = config
        self.model = LPQwen3MoeModel(config)
        # self.lm_head = KLinear(self.lm_head, op="KLinearMarlin")
        # self.lm_head.device = "cpu"

    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        
        cuda_device = torch.device('cuda', torch.cuda.current_device())

        if input_ids.shape[1] > 50000:
            ring_buffer_len = 2
        else:
            ring_buffer_len = self.model.config.num_experts

        print(f"CONTEXT_LEN={input_ids.shape[1]}, ADOPT EXP BUFFER SIZE={ring_buffer_len}")

        LPQwen3MoeSparseMoeBlock.ring_buffer_mgr.reset({
            "gate": torch.empty((ring_buffer_len, self.config.moe_intermediate_size, self.config.hidden_size), dtype=self.config.weight_dtype, device=cuda_device),
            "up": torch.empty((ring_buffer_len, self.config.moe_intermediate_size, self.config.hidden_size), dtype=self.config.weight_dtype, device=cuda_device),
            "down": torch.empty((ring_buffer_len, self.config.hidden_size, self.config.moe_intermediate_size), dtype=self.config.weight_dtype, device=cuda_device),
        }, ring_buffer_len)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        StreamLoader.load()

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        LPQwen3MoeSparseMoeBlock.ring_buffer_mgr.to_("cpu")

        hidden_states = outputs[0]

        self.lm_head = self.lm_head.to(device=cuda_device)
        logits = self.lm_head(hidden_states[:,-1:,:])
        self.lm_head = self.lm_head.to("cpu")

        loss = None
        aux_loss = None

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def clear_cuda(self):
        self.lm_head = self.lm_head.to("cpu")
        self.model.embed_tokens = self.model.embed_tokens.to("cpu")
