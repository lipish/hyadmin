import torch
from torch import nn

from heyi.models.qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoeDecoderLayer,
    Qwen3MoeConfig,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeRotaryEmbedding,
)

from heyi.operators import (
    KQwen3MoeSparseMoeBlock,
    GQA,
    FusedRMSNorm,
    KLinear,
    KExpertsCPU
)



class OptQwen3MoeAttention(GQA):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        linear_op = "KLinearFP8" if config.weight_dtype == torch.float8_e4m3fn else "KLinearTorch"
        self.q_proj = KLinear(self.q_proj, op=linear_op)
        self.k_proj = KLinear(self.k_proj, op=linear_op)
        self.v_proj = KLinear(self.v_proj, op=linear_op)
        self.o_proj = KLinear(self.o_proj, op=linear_op)

        # self.q_norm = FusedRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        # self.k_norm = FusedRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def to_(self, device: str):
        self.q_proj.to_(device)
        self.k_proj.to_(device)
        self.v_proj.to_(device)
        self.o_proj.to_(device)


class OptQwen3MoeMLP(Qwen3MoeMLP):
    def __init__(self, config):
        super().__init__(config)
        op = "KLinearFP8" if config.weight_dtype == torch.float8_e4m3fn else "KLinearTorch"
        self.gate_proj = KLinear(self.gate_proj, op=op)
        self.up_proj = KLinear(self.up_proj, op=op)
        self.down_proj = KLinear(self.down_proj, op=op)

    def to_(self, device: str):
        self.gate_proj.to_(device)
        self.up_proj.to_(device)
        self.down_proj.to_(device)


class OptQwen3MoeSparseMoeBlock(KQwen3MoeSparseMoeBlock):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.experts = KExpertsCPU(config).cpu()


class OptQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = OptQwen3MoeAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = OptQwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        # self.input_layernorm = FusedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = FusedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class OptQwen3MoeModel(Qwen3MoeModel):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)

        self.layers = nn.ModuleList([
            OptQwen3MoeDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
            # for layer_idx in range(1)
        ])
        # self.norm = FusedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        with torch.device("cuda"):
            self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)


class OptQwen3MoeForCausalLM(Qwen3MoeForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = OptQwen3MoeModel(config)
        # self.lm_head = KLinear(self.lm_head, op="KLinearMarlin")
        # self.lm_head.device = "cpu"

    def to_(self, device: str):
        self.lm_head = self.lm_head.to(device=device, non_blocking=True)
        self.model.embed_tokens = self.model.embed_tokens.to(device=device, non_blocking=True)
        for layer in self.model.layers:
            layer.self_attn.to_(device)
            if isinstance(layer.mlp, OptQwen3MoeMLP):
                layer.mlp.to_(device)
            else:
                assert isinstance(layer.mlp, OptQwen3MoeSparseMoeBlock)

