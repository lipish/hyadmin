import torch
import flashinfer
from .decode_runner import DecodeRunner


class MLADecodeRunner(DecodeRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            self.float_workspace_buffer, use_cuda_graph=False
        )
        with torch.no_grad():
            self.warmup_and_capture_graph()

    def wrapper_plan_decode1(self, B: int):
        self.wrapper.plan(
            torch.arange(0, B + 1, dtype=torch.int32, device="cuda"),
            self.kvcache.buffers["page_indptr"],
            self.kvcache.buffers["page_indices"],
            self.input_buffer[B]["cache_position"].to(torch.int32) + 1,
            num_heads=self.model.config.num_attention_heads,
            head_dim_ckv=self.model.config.kv_lora_rank,
            head_dim_kpe=self.model.config.qk_rope_head_dim,
            page_size=self.kvcache.page_size,
            causal=True,
            sm_scale=self.model.model.layers[0].self_attn.softmax_scale,  # type: ignore
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

    def wrapper_plan_cprefill(self, qo_indptr: torch.Tensor, kv_len_arr: torch.Tensor):
        self.wrapper.plan(
            qo_indptr,
            self.kvcache.buffers["page_indptr"],
            self.kvcache.buffers["page_indices"],
            kv_len_arr,
            num_heads=self.model.config.num_attention_heads,
            head_dim_ckv=self.model.config.kv_lora_rank,
            head_dim_kpe=self.model.config.qk_rope_head_dim,
            page_size=self.kvcache.page_size,
            causal=True,
            sm_scale=self.model.model.layers[0].self_attn.softmax_scale,  # type: ignore
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
