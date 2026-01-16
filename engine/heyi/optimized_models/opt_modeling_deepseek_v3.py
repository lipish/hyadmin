
from torch import nn
from typing import Optional
from transformers.configuration_utils import PretrainedConfig

from heyi.models.deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3DecoderLayer,
    DeepseekV3Config,
    DeepseekV3MLP,
    MoEGate
)

from heyi.operators import (
    KDeepseekV2Model, 
    KLinear,
    FusedRMSNorm as OptDeepseekV3RMSNorm,
    MLA,
    YarnRotaryEmbeddingV3 as OptDeepseekV3YarnRotaryEmbedding,
    KDeepseekV3MoE,
    KExpertsCPU,
    KMoEGate
)

class OptDeepseekV3Attention(MLA):
    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.device = "cuda:0"
        self.q_a_proj = KLinear(self.q_a_proj, op="KLinearFP8")
        self.q_b_proj = KLinear(self.q_b_proj, op="KLinearFP8")
        self.kv_a_proj_with_mqa = KLinear(self.kv_a_proj_with_mqa, op="KLinearFP8")
        self.kv_b_proj.dequantize = True
        self.o_proj = KLinear(self.o_proj, op="KLinearFP8")

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
        self.rotary_emb = OptDeepseekV3YarnRotaryEmbedding(config)
        self.absorb_for_prefill = True

    def to_(self, device: str):
        self.q_b_proj.to_(device)
        self.kv_a_proj_with_mqa.to_(device)
        self.q_a_proj.to_(device)
        self.kv_b_proj = self.kv_b_proj.to(device, non_blocking=True)
        self.o_proj.to_(device)

class OptDeepseekV3MLP(DeepseekV3MLP):
    def __init__(self, config, op="KLinearFP8"):
        super().__init__(config)
        self.gate_proj = KLinear(self.gate_proj, op=op)
        self.up_proj = KLinear(self.up_proj, op=op)
        self.down_proj = KLinear(self.down_proj, op=op)

    def to_(self, device: str):
        self.gate_proj.to_(device)
        self.up_proj.to_(device)
        self.down_proj.to_(device)


class OptDeepseekV3MoE(KDeepseekV3MoE):
    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.experts = KExpertsCPU(config).cpu()
        # self.gate = KMoEGate(config)
        self.shared_experts = OptDeepseekV3MLP(config)


class OptDeepseekV3DecoderLayer(DeepseekV3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = OptDeepseekV3Attention(config, layer_idx)
        self.mlp = (
            OptDeepseekV3MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else OptDeepseekV3MLP(config)
        )
        self.input_layernorm = OptDeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = OptDeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

class OptDeepseekV3Model(KDeepseekV2Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                OptDeepseekV3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
                # for layer_idx in range(4)
            ]
        )
        # self.norm = OptDeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.embed_tokens.device = "cpu"


class OptDeepseekV3ForCausalLM(DeepseekV3ForCausalLM):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.model = OptDeepseekV3Model(self.model.config)
        # self.lm_head = KLinear(self.lm_head, op="KLinearMarlin")
        # self.lm_head.device = "cpu"

    def to_(self, device: str):
        self.lm_head = self.lm_head.to(device=device, non_blocking=True)
        self.model.embed_tokens = self.model.embed_tokens.to(device=device, non_blocking=True)
        for layer in self.model.layers:
            layer.self_attn.to_(device)
            if isinstance(layer.mlp, OptDeepseekV3MLP):
                layer.mlp.to_(device)
            else:
                assert isinstance(layer.mlp, OptDeepseekV3MoE)
                layer.mlp.shared_experts.to_(device)
