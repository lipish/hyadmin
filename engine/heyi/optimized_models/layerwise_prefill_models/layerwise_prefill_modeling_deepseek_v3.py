from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from heyi.models.deepseek_v3 import (
    DeepseekV3Config,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MoE,
)
from heyi.operators import MLA
from heyi.operators import FusedRMSNorm as OptDeepseekV3RMSNorm
from heyi.operators import KDeepseekV2Model, YarnRotaryEmbeddingV3
from heyi.operators.mla import apply_rotary_pos_emb
from heyi.config import Config
from heyi.utils.kvcache.kvcache import PagedMLACache
from heyi.utils.log import logger
from heyi.utils.ring_buffer import RingBufferMgr
from heyi.utils.stream_loader import StreamLoader, register_opevs

from .common import LP_LinearFP8_FromGPU, LP_MLP_FromCPUWithBuffer, LP_MLP_FromGPU

# from line_profiler import profile as lprofile

class LPYarnRotaryEmbedding(YarnRotaryEmbeddingV3):
    def __init__(
        self,
        config: PretrainedConfig,
        device: int
    ):
        super().__init__(config, device)
        self.config = config
        self.device = device
        with torch.device(device), torch.cuda.device(device):
            self.load()

class LPDeepseekV3Attention(MLA):
    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.q_a_proj = LP_LinearFP8_FromGPU(self.q_a_proj)
        self.q_b_proj = LP_LinearFP8_FromGPU(self.q_b_proj)
        self.kv_a_proj_with_mqa = LP_LinearFP8_FromGPU(self.kv_a_proj_with_mqa)
        self.kv_b_proj.dequantize = True
        self.o_proj = LP_LinearFP8_FromGPU(self.o_proj)

        self.q_a_layernorm = OptDeepseekV3RMSNorm(config.q_lora_rank)
        self.kv_a_layernorm = OptDeepseekV3RMSNorm(config.kv_lora_rank)

        # kwargs = {
        #     key: self.config.rope_scaling[key]
        #     for key in [
        #         "original_max_position_embeddings",
        #         "beta_fast",
        #         "beta_slow",
        #         "mscale",
        #         "mscale_all_dim",
        #     ]
        #     if key in self.config.rope_scaling
        # }
        # scaling_factor = self.config.rope_scaling["factor"]
        self.rotary_emb = LPYarnRotaryEmbedding(config, Config().layerwise_prefill_device)

    @torch.no_grad()
    # @torch.cuda.nvtx.range("ATTN FWD")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[PagedMLACache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        q = hidden_states.new_empty(bsz, q_len, self.num_heads * self.q_head_dim)
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            CHUNK_SIZE = 65536
            for l in range(0, q_len, CHUNK_SIZE):
                r = min(l + CHUNK_SIZE, q_len)
                q[:, l:r] = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states[:, l:r])))

            # q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
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
        q_pe_, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, None, unsqueeze_dim=2)
        q_pe.copy_(q_pe_)
        # q_pe [bsz, q_len, self.num_heads, self.qk_rope_head_dim] k_pe [bsz, q_len, 1, self.qk_rope_head_dim]

        if past_key_value is not None:
            k_pe.squeeze(0)
            compressed_kv.squeeze(0)

            with torch.cuda.device(0):
                compressed_kv_with_k_pe = past_key_value.update(compressed_kv.to(0), k_pe.to(0), self.layer_idx)

            compressed_kv_with_k_pe = compressed_kv_with_k_pe[past_key_value.buffers["page_indices"]].to(Config().layerwise_prefill_device)
            compressed_kv, k_pe = torch.split(
                compressed_kv_with_k_pe, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
        compressed_kv = compressed_kv.view(bsz, -1, self.kv_lora_rank)
        compressed_kv = compressed_kv[:, :kv_seq_len]

        k_pe_v = k_pe.new_empty(
            bsz, kv_seq_len, self.num_heads, self.q_head_dim + self.v_head_dim
        )

        CHUNK_SIZE = 32768
        for l in range(0, kv_seq_len, CHUNK_SIZE):
            r = min(l + CHUNK_SIZE, kv_seq_len)
            kv_slice = self.kv_b_proj(compressed_kv[:, l:r]).view(
                bsz, r - l, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_pe_v[:, l:r, ..., :self.qk_nope_head_dim] = kv_slice[..., :self.qk_nope_head_dim]
            k_pe_v[:, l:r, ..., -self.v_head_dim:] = kv_slice[..., self.qk_nope_head_dim:]
            del kv_slice

        k_pe = k_pe.view(bsz, -1, self.qk_rope_head_dim)[:, :kv_seq_len]
        k_pe_v[..., self.qk_nope_head_dim : self.qk_nope_head_dim + self.qk_rope_head_dim] = k_pe.view(bsz, kv_seq_len, 1, -1)

        k = k_pe_v[..., :self.q_head_dim]
        v = k_pe_v[..., self.q_head_dim:]

        del k_pe, compressed_kv, compressed_kv_with_k_pe

        attn_wrapper = kwargs.get("attn_wrapper")
        attn_output = attn_wrapper.run(
            q.view(-1, *q.shape[2:]),
            k.view(-1, *k.shape[2:]),
            v.view(-1, *v.shape[2:]),
        )
        del q, k, v, k_pe_v

        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[..., : self.v_head_dim]

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    
    def on(self):
        self.q_a_proj.on()
        self.q_b_proj.on()
        self.kv_a_proj_with_mqa.on()
        self.kv_b_proj = self.kv_b_proj.to(
            torch.cuda.current_device(), non_blocking=True
        )
        self.o_proj.on()

    def off(self):
        self.q_a_proj.off()
        self.q_b_proj.off()
        self.kv_a_proj_with_mqa.off()
        self.kv_b_proj = self.kv_b_proj.cpu()
        self.o_proj.off()


class LPDeepseekV3MoE(DeepseekV3MoE):
    ring_buffer_mgr = RingBufferMgr()

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.experts = nn.ModuleList(
            [
                LP_MLP_FromCPUWithBuffer(
                    config, 
                    intermediate_size=config.moe_intermediate_size,
                    ring_buffer_mgr=LPDeepseekV3MoE.ring_buffer_mgr,
                )
                for i in range(config.n_routed_experts)
            ]
        )
        # self.gate = KMoEGate(config)
        self.shared_experts = LP_MLP_FromGPU(config)
        self.experts_opevs = register_opevs([expert for expert in self.experts])
        self.shared_expert_opevs = register_opevs([self.shared_experts])[0]

    # @torch.profiler.record_function("moe fwd")
    # @torch.cuda.nvtx.range("MoE FWD")
    # @lprofile
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            # print(f"[MODEL] WAIT FOR SHEXP{self.shared_experts.key} EV_ON")
            # torch.cuda.nvtx.range_push(
            #     f"[MODEL] WAIT FOR SHEXP {self.shared_experts.key} EV_ON"
            # )
            self.shared_expert_opevs.ev.wait_on()
            # torch.cuda.nvtx.range_pop()
            # print(f"[MODEL] CONFIRMED {self.shared_experts.key} EV_ON")
            y.add_(self.shared_experts(identity))
            # print(f"[MODEL] SET SHEXP {self.shared_experts.key} EV_OFF")
            self.shared_expert_opevs.ev.set_off()
        return y

    @torch.no_grad()
    # @lprofile
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        # sorted_tokens = x[idxs // topk_ids.shape[1]]
        # torch.cuda.nvtx.range_push("tokens_per_expert.cpu()")
        tokens_per_expert = tokens_per_expert.cpu()
        # torch.cuda.nvtx.range_pop()

        new_x = x.new_empty(topk_ids.shape[0] * topk_ids.shape[1], x.shape[-1])
        start_idx = 0
        # self.opev.ev.wait_on()
        for i, num_tokens in enumerate(tokens_per_expert):
            # print(f"[MODEL] EXPERT # {i}")
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                # print(f"[MODEL] NOT USED: SET expert.{i} EV_OFF")
                self.experts_opevs[i].ev.set_off()
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]

            # print(f"[MODEL] WAIT FOR EXP {expert.key} EV_ON")
            # torch.cuda.nvtx.range_push(f"[MODEL] WAIT FOR {expert.key} EV_ON")
            self.experts_opevs[i].ev.wait_on()
            # torch.cuda.nvtx.range_pop()
            # print(f"[MODEL] CONFIRMED {expert.key} EV_ON")

            tokens_for_this_expert = x[idxs[start_idx:end_idx] // topk_ids.shape[1]]
            new_x[idxs[start_idx:end_idx]] = expert(tokens_for_this_expert)

            # print(f"[MODEL] SET EXP {expert.key} EV_OFF")
            self.experts_opevs[i].ev.set_off()

            start_idx = end_idx

        # new_x = torch.empty_like(outs)
        # new_x[idxs] = outs
        # del outs

        # final_out = (
        #     new_x.view(*topk_ids.shape, -1)
        #     .type(topk_weight.dtype)
        #     .mul_(topk_weight.unsqueeze(dim=-1))
        #     .sum(dim=1)
        #     .type(new_x.dtype)
        # )

        new_x = new_x.view(*topk_ids.shape, -1)
        topk_weight = topk_weight.unsqueeze(dim=-1)
        seqlen = topk_ids.shape[0]

        CHUNK_SIZE = 16384

        final_out = torch.empty_like(x)
        for l in range(0, new_x.shape[0], CHUNK_SIZE):
            r = min(l + CHUNK_SIZE, seqlen)
            final_out[l:r] = (
                new_x[l:r]
                .type(topk_weight.dtype)
                .mul_(topk_weight[l:r])
                .sum(dim=1)
                .type(new_x.dtype)
            )
        # self.opev.ev.set_off()
        return final_out

class LPDeepseekV3DecoderLayer(DeepseekV3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = LPDeepseekV3Attention(config, layer_idx)
        attn_opev = register_opevs([self.self_attn])
        self.mlp = (
            LPDeepseekV3MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else LP_MLP_FromGPU(config)
        )
        mlp_opev = register_opevs([self.mlp]) if isinstance(self.mlp, LP_MLP_FromGPU) else []
        self.stream_load_opevs = attn_opev + mlp_opev
        self.input_layernorm = OptDeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = OptDeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    # @lprofile
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
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

        # self_attn
        hidden_states[:], self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
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

        if isinstance(self.mlp, DeepseekV3MoE):
            pass
        else:
            # print(f"[MODEL]: WAIT FOR MLP {self.mlp.key} EV_ON")
            self.stream_load_opevs[1].ev.wait_on()
            # print(f"[MODEL]: CONFIRMED {self.mlp.key} EV_ON")

        hidden_states.copy_(self.mlp(hidden_states))
        torch.cuda.synchronize()

        if isinstance(self.mlp, DeepseekV3MoE):
            pass
        else:
            # print(f"[MODEL]: SET {self.mlp.key} EV_OFF")
            self.stream_load_opevs[1].ev.set_off()

        if offload_residuals:
            residual = residual.to(device, non_blocking=True)
        # hidden_states = residual + hidden_states
        hidden_states.add_(residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LPDeepseekV3Model(KDeepseekV2Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                LPDeepseekV3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
                # for layer_idx in range(4)
            ]
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[PagedMLACache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attn_wrapper = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if inputs_embeds is None:
            cuda_device = torch.device('cuda', torch.cuda.current_device())
            self.embed_tokens = self.embed_tokens.to(cuda_device)
            inputs_embeds = self.embed_tokens(input_ids)  # .to(org_device)
            self.embed_tokens = self.embed_tokens.to("cpu")
            # input_ids = input_ids.to(org_device)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = None

        # embed positions
        hidden_states = inputs_embeds
        del inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        import time
        base = time.perf_counter()
        last_layer_timestamp = time.perf_counter()
        for i, decoder_layer in enumerate(self.layers):
            # torch.cuda.nvtx.range_push("[MODEL] ONLOAD KVCache")
            if Config().layerwise_prefill_device == 0:
                past_key_values.to_(0, i)
            # torch.cuda.nvtx.range_pop()
            assert isinstance(decoder_layer, LPDeepseekV3DecoderLayer)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # with open("log.txt", "a") as f:
            #     f.write(f"@@@@@@@@@@@@@@@@@layer {i}@@@@@@@@@@@@@@@@@@@@ \n")
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                attn_wrapper=attn_wrapper,
            )
            hidden_states = layer_outputs[0]

            # @@@@@@@ TODO open this notes, tmp close to fit deepseekv3
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            # torch.cuda.nvtx.range_push("[MODEL] OFFLOAD KVCache")
            if Config().layerwise_prefill_device == 0:
                past_key_values.to_("cpu", i)
            # torch.cuda.nvtx.range_pop()
            print(f"[MODEL] layer {i}: {time.perf_counter() - last_layer_timestamp:.03f}s")
            last_layer_timestamp = time.perf_counter()

        hidden_states = self.norm(hidden_states).bfloat16()
        logger.info(f"[MODEL] all layers: {time.perf_counter() - base:.03f}s")

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LPDeepseekV3ForCausalLM(DeepseekV3ForCausalLM):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.model = LPDeepseekV3Model(self.model.config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attn_wrapper = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # import nvtx
        # nvtx.push_range("LPREFILL")

        cuda_device = torch.device('cuda', torch.cuda.current_device())
        
        if input_ids.shape[1] > 50000:
            ring_buffer_len = 2
        else:
            ring_buffer_len = self.model.config.n_routed_experts
        # ring_buffer_len = 1

        print(f"CONTEXT_LEN={input_ids.shape[1]}, ADOPT EXP BUFFER SIZE={ring_buffer_len}")

        LPDeepseekV3MoE.ring_buffer_mgr.reset({
            "gate": torch.empty((ring_buffer_len, 2048, 7168), dtype=torch.float8_e4m3fn, device=cuda_device),
            "up": torch.empty((ring_buffer_len, 2048, 7168), dtype=torch.float8_e4m3fn, device=cuda_device),
            "down": torch.empty((ring_buffer_len, 7168, 2048), dtype=torch.float8_e4m3fn, device=cuda_device),
        }, ring_buffer_len)

        # LPDeepseekV3MoE.ring_buffer_mgr.to_(cuda_device)        

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        StreamLoader.load()

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            attn_wrapper=attn_wrapper,
        )
        LPDeepseekV3MoE.ring_buffer_mgr.to_("cpu")

        hidden_states = outputs[0]
        
        self.lm_head = self.lm_head.to(device=cuda_device)
        logits = self.lm_head(hidden_states[:,-1:,:])
        logits = logits.float()
        self.lm_head = self.lm_head.to("cpu")

        loss = None
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        # nvtx.pop_range()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def clear_cuda(self):
        self.lm_head = self.lm_head.to("cpu")
        self.model.embed_tokens = self.model.embed_tokens.to("cpu")
        for layer in self.model.layers:
            layer.self_attn.kv_b_proj = layer.self_attn.kv_b_proj.to("cpu")
