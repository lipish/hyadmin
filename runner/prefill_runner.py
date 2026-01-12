import torch
from transformers.modeling_utils import PreTrainedModel

import flashinfer
from heyi.utils.kvcache.kvcache import PagedMLACache, PagedGQACache, PagedKVCache
from heyi.utils.request import Request

from .base_runner import BaseRunner
from abc import abstractmethod
from typing import Union


class PrefillRunner(BaseRunner):
    def __init__(
        self,
        runner_id: int,
        model: PreTrainedModel,
        kvcache: PagedKVCache,
        device: int,
    ):
        super().__init__(runner_id, model, kvcache)
        self.req = None
        self.done = False
        self.device = device
        self.workspace_buffer = torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.wrapper: flashinfer.BatchPrefillWithRaggedKVCacheWrapper = (
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer, kv_layout="NHD", backend="auto"
            )
        )

        # self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        # self.wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(self.workspace_buffer, kv_layout="NHD", backend="auto")

    def _wrapper_plan(self, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor):
        raise NotImplementedError

    @torch.no_grad
    def prefill(self, req: Request):
        matches = self.kvcache.match(req.all_ids[:, : req.all_length])
        req.prefilled_length = matches[0].len * self.kvcache.page_size

        # print()
        # print("-" * 30, "PREFILL PLAN", "-" * 30)
        # print()

        # print(self.kvcache.prefix_tree)
        # print(f"MATCH {req.all_ids[:, :req.all_length]}")
        # print(f"MATCHED {matches[0]}, {req.prefilled_length=}")

        input_ids, cache_position = req.next_full_prefill()
        print(f"do prefill: {input_ids.shape=}, {cache_position.shape}")
        req.matches = self.kvcache.plan(
            matches,
            req.all_ids[:, : cache_position[-1].item() + 1],
            return_matches=True,
        )

        bsz, q_len = input_ids.shape
        qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=self.device)
        kv_indptr = torch.tensor([0, req.prompt_length], dtype=torch.int32, device=self.device)

        print("LP: plan")
        self._wrapper_plan(qo_indptr, kv_indptr)

        # print(f"PLAN {req.all_ids[:, : cache_position[-1].item() + 1]}")
        # print(self.kvcache.prefix_tree)
        # print(self.kvcache)

        # print()
        # print("-" * 80)
        # print()

        # inputs_embeds = self.model.model.embed_tokens(input_ids)
        print("LP: run")
        with torch.no_grad(), torch.cuda.device(self.device):
            logits = self.model(
                input_ids=input_ids.cuda(),
                cache_position=cache_position.cuda(),
                past_key_values=self.kvcache,
                return_dict=False,
                use_cache=True,
                attn_wrapper=self.wrapper
            )[0].squeeze(dim=1)
        return logits

class MLA_LPrefillRunner(PrefillRunner):
    def _wrapper_plan(self, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor):
        num_qo_heads = num_kv_heads = self.model.config.num_attention_heads
        head_dim_qk = self.model.config.qk_nope_head_dim + self.model.config.qk_rope_head_dim
        head_dim_vo = self.model.config.qk_nope_head_dim

        self.wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            sm_scale=self.model.model.layers[0].self_attn.softmax_scale,
        )

class GQA_LPrefillRunner(PrefillRunner):
    # wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper
    # def _wrapper_plan(self, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor):
    #     num_qo_heads = self.model.config.num_attention_heads
    #     num_kv_heads = self.model.config.num_key_value_heads
    #     head_dim_qk = self.model.model.layers[0].self_attn.head_dim

    #     B = 1
    #     page_indptr = self.kvcache.buffers["page_indptr"][:B + 1]
    #     page_indices = self.kvcache.buffers["page_indices"][:page_indptr[-1]]
    #     last_page_len = self.kvcache.buffers["last_page_len"][:B]
    #     self.wrapper.plan(
    #         qo_indptr,
    #         page_indptr,
    #         page_indices,
    #         last_page_len,
    #         num_qo_heads=num_qo_heads,
    #         num_kv_heads=num_kv_heads,
    #         head_dim_qk=head_dim_qk,
    #         page_size=self.kvcache.page_size,
    #         causal=True,
    #         q_data_type=torch.bfloat16,
    #         kv_data_type=torch.bfloat16,
    #     )

    def _wrapper_plan(self, qo_indptr: torch.Tensor, kv_indptr: torch.Tensor):
        num_qo_heads = self.model.config.num_attention_heads
        num_kv_heads = self.model.config.num_key_value_heads
        head_dim_qk = self.model.model.layers[0].self_attn.head_dim
        self.wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=None,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            sm_scale=self.model.model.layers[0].self_attn.scaling,
        )
