from abc import abstractmethod
from typing import Dict, List, Union

import flashinfer
import torch
from torch.nn.attention import SDPBackend
from transformers.modeling_utils import PreTrainedModel

from heyi.utils.kvcache.kvcache import PagedMLACache
from heyi.utils.log import logger
from heyi.utils.request import DecodeBatch, Request

from .base_runner import BaseRunner


class DecodeRunner(BaseRunner):
    def __init__(
        self,
        runner_id: int,
        model: PreTrainedModel,
        kvcache: PagedMLACache,
        Bs: List[int],
        use_cuda_graph: bool,
    ):
        super().__init__(runner_id, model, kvcache)
        self.stream = torch.cuda.Stream()
        self.cuda_graph: Dict[int, torch.cuda.CUDAGraph] = {}

        self.Bs = Bs
        self.maxB = max(Bs)
        self.input_buffer = {
            B: {
                "input_ids": torch.zeros((B, 1), dtype=torch.long).cuda(),
                "past_key_values": self.kvcache,
                "cache_position": torch.zeros((B), dtype=torch.int, device=0),
            }
            for B in self.Bs
        }
        self.output_buffer = {
            "logits": torch.empty((self.maxB, model.config.vocab_size), dtype=torch.float32, device=0)
        }

        self.float_workspace_buffer = torch.empty(128*1024*1024, dtype=torch.uint8, device=0)
        self.wrapper: Union[
            flashinfer.BatchDecodeMlaWithPagedKVCacheWrapper,
            flashinfer.BatchPrefillWithPagedKVCacheWrapper
        ] = None # type: ignore

        self.use_cuda_graph = use_cuda_graph
    
    @abstractmethod
    def wrapper_plan_decode1(self, B: int):
        raise NotImplementedError
    
    @abstractmethod
    def wrapper_plan_cprefill(self, qo_indptr: torch.Tensor, kv_len_arr: torch.Tensor):
        raise NotImplementedError

    def warmup_and_capture_graph(self):
        for B in self.Bs:
            self._warmup_and_capture_graph(B)

    @torch.no_grad()
    def _warmup_and_capture_graph(self, B: int):
        logger.info(f"Rnr#{self.runner_id}: warmup {B=}...")
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream): # type: ignore
            for _ in range(3):
                self.kvcache.plan(
                    self.kvcache.match(self.input_buffer[B]["input_ids"]),
                    self.input_buffer[B]["input_ids"],
                )
                self.wrapper_plan_decode1(B)
                _ = self.model(
                    input_ids=self.input_buffer[B]["input_ids"],
                    past_key_values=self.input_buffer[B]["past_key_values"],
                    cache_position=self.input_buffer[B]["cache_position"],
                    use_cache=True,
                    attn_wrapper=self.wrapper,
                )
        torch.cuda.current_stream().wait_stream(self.stream)

        if not self.use_cuda_graph:
            return

        self.cuda_graph[B] = torch.cuda.CUDAGraph()
        logger.info(f"Rnr#{self.runner_id}: capturing {B=}...")
        self.kvcache.plan(
            self.kvcache.match(self.input_buffer[B]["input_ids"]),
            self.input_buffer[B]["input_ids"],
        )
        self.wrapper_plan_decode1(B)
        with torch.cuda.graph(cuda_graph=self.cuda_graph[B], stream=self.stream):
            self.output_buffer["logits"][:B].copy_(self.model(
                input_ids=self.input_buffer[B]["input_ids"],
                past_key_values=self.input_buffer[B]["past_key_values"],
                cache_position=self.input_buffer[B]["cache_position"],
                use_cache=True,
                attn_wrapper=self.wrapper,
            )[0].squeeze(dim=1))
        torch.cuda.synchronize("cuda")

    @torch.no_grad
    def prefill_chunk(self, req: Request):
        '''Currently only support per request chunked prefill'''
        if not req.matches:
            req.matches = self.kvcache.match(req.all_ids[:, : req.all_length])
        req.prefilled_length = req.matches[0].len * self.kvcache.page_size

        # print()
        # print("-" * 30, "PREFILL PLAN", "-" * 30)
        # print()

        # print(self.kvcache.prefix_tree)
        # print(f"MATCH {req.all_ids[:, :req.all_length]}")
        # print(f"MATCHED {matches[0]}, {req.prefilled_length=}")

        input_ids, cache_position = req.next_chunk()
        req.matches = self.kvcache.plan(
            req.matches,
            req.all_ids[:, : cache_position[-1].item() + 1],
            return_matches=True,
        )

        # print(f"PLAN {req.all_ids[:, : cache_position[-1].item() + 1]}")
        # print(self.kvcache.prefix_tree)
        # print(self.kvcache)

        # print()
        # print("-" * 80)
        # print()

        # inputs_embeds = self.model.model.embed_tokens(input_ids)
        bsz, q_len = input_ids.shape
        qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
        kv_len_arr = torch.tensor(
            [cache_position[-1].item() + 1], dtype=torch.int32, device="cuda"
        )
        self.wrapper_plan_cprefill(qo_indptr, kv_len_arr)

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                cache_position=cache_position,
                past_key_values=self.kvcache,
                return_dict=False,
                use_cache=True,
                attn_wrapper=self.wrapper,
            )[0].squeeze(dim=1)
        return logits

    @torch.no_grad
    def plan_decode1(self, batch: DecodeBatch):
        B = batch.B
        batch.decode_runner_id = self.runner_id

        input_ids, cache_position = batch.next_token()
        self.input_buffer[B]["input_ids"].copy_(input_ids)
        self.input_buffer[B]["cache_position"].copy_(cache_position)

        assert batch.matches
        batch.matches = self.kvcache.plan(
            batch.matches, batch.all_ids, return_matches=True
        )
        self.wrapper_plan_decode1(B)

    @torch.no_grad
    def decode1(self, batch: DecodeBatch):
        B = batch.B

        with torch.nn.attention.sdpa_kernel(
            backends=[
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
        ):
            if self.use_cuda_graph:
                with torch.cuda.stream(self.stream): # type: ignore
                    self.cuda_graph[B].replay()
                self.stream.synchronize()
            else:
                with torch.no_grad():
                    self.output_buffer["logits"][:B].copy_(self.model(
                        input_ids=self.input_buffer[B]["input_ids"],
                        past_key_values=self.input_buffer[B]["past_key_values"],
                        cache_position=self.input_buffer[B]["cache_position"],
                        use_cache=True,
                        attn_wrapper=self.wrapper,
                    )[0].squeeze(dim=1))

            logits = self.output_buffer["logits"][:B]
            return logits
