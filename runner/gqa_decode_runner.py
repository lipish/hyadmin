import torch
import flashinfer
from .decode_runner import DecodeRunner


class GQADecodeRunner(DecodeRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.float_workspace_buffer = torch.empty(
            384 * 1024 * 1024, dtype=torch.uint8, device=0
        )
        self.decode_wrappers = {
            B: flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.float_workspace_buffer,
                use_cuda_graph=True,
                qo_indptr_buf=self.kvcache.buffers["qo_indptr"][:B + 1],
                paged_kv_indptr_buf=self.kvcache.buffers["page_indptr"][:B + 1],
                paged_kv_indices_buf=self.kvcache.buffers["page_indices"],
                paged_kv_last_page_len_buf=self.kvcache.buffers["last_page_len"][:B],
            )
            for B in self.Bs
        }
        self.prefill_wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper = (
            flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.float_workspace_buffer,
                use_cuda_graph=False,
            )
        )
        with torch.no_grad():
            self.warmup_and_capture_graph()

    def wrapper_plan_decode1(self, B: int):
        qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
        page_indptr = self.kvcache.buffers["page_indptr"][:B + 1]
        page_indices = self.kvcache.buffers["page_indices"][:page_indptr[-1]]
        last_page_len = self.kvcache.buffers["last_page_len"][:B]
        self.wrapper = self.decode_wrappers[B]
        self.wrapper.plan(
            qo_indptr,            
            page_indptr,
            page_indices,
            last_page_len,
            num_qo_heads=self.model.config.num_attention_heads,
            num_kv_heads=self.model.config.num_key_value_heads,
            head_dim_qk=self.model.model.layers[0].self_attn.head_dim,  # type: ignore
            page_size=self.kvcache.page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

    def wrapper_plan_cprefill(self, qo_indptr: torch.Tensor, kv_len_arr: torch.Tensor):
        B = qo_indptr.shape[0] - 1
        page_indptr = self.kvcache.buffers["page_indptr"][:B + 1]
        page_indices = self.kvcache.buffers["page_indices"][:page_indptr[-1]]
        last_page_len = self.kvcache.buffers["last_page_len"][:B]
        self.wrapper = self.prefill_wrapper
        self.wrapper.plan(
            qo_indptr,
            page_indptr,
            page_indices,
            last_page_len,
            num_qo_heads=self.model.config.num_attention_heads,
            num_kv_heads=self.model.config.num_key_value_heads,
            head_dim_qk=self.model.model.layers[0].self_attn.head_dim,  # type: ignore
            page_size=self.kvcache.page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
