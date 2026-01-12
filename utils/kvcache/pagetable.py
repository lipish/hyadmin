import heapq
import threading
from typing import List, Optional

import flashinfer
import torch
from transformers.configuration_utils import PretrainedConfig


class PageTable:
    """
    global states:
    - page_filled_len:      [max_num_pages], 0 ~ page_size, tracks how much of each page is filled
    - mla_paged_kv_cache:   [num_layers, max_num_pages, page_size, ckv_dim + kpe_dim], the actual KV cache storage
    - free_pages:           set of free page IDs
    - used_pages:           set of used page IDs

    per-step states (passed as parameters to update method):
    - kv_page_indices:      [max_num_pages], flat ragged list of page IDs
    - kv_page_indptr:       [B + 1], marks the start of each request's pages
    - kv_last_page_len:     [B], length of the final page for each request
    """

    def __init__(
        self,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
        pages: List[torch.Tensor],
        device: int = 0,
    ):
        self.lock = threading.Lock()
        self.max_batch_size = max_batch_size
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.pages = pages
        self.device = device

        self.free_pages = set([i for i in range(max_num_pages)])
        self.used_pages = set()

        self.page_filled_len = torch.zeros(self.max_num_pages, dtype=torch.int32)

    @property
    def n_free_pages(self):
        return len(self.free_pages)

    def allocate(self, num_pages: int) -> List[int]:
        """Vectorized allocation of the first num_pages free pages."""
        with self.lock:
            # print(f"Allocating {num_pages} pages")
            assert self.n_free_pages >= num_pages
            alloc_pages = [self.free_pages.pop() for _ in range(num_pages)]
            for page in alloc_pages:
                self.used_pages.add(page)
        return alloc_pages

    def free(self, pages_to_free: List[int]):
        with self.lock:
            for page in pages_to_free:
                self.used_pages.remove(page)
                self.free_pages.add(page)
            self.page_filled_len[pages_to_free] = 0

    def set_page_filled_len(self, page_indices: torch.Tensor, lengths: torch.Tensor):
        with self.lock:
            self.page_filled_len[page_indices] = lengths

    def to_(self, device, layer: Optional[int] = None):
        if layer is None:
            for i, kvc in enumerate(self.pages):
                self.pages[i] = kvc.to(device=device, non_blocking=True)
        else:
            self.pages[layer] = self.pages[layer].to(
                device=device, non_blocking=True
            )

    def __str__(self):
        return (
            f"{self.n_free_pages=}\n"
            f"{self.free_pages=}\n"
            f"{self.used_pages=}\n"
            f"{self.page_filled_len=}"
        )

class MLAPageTable(PageTable):
    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
        device: str = "cuda",
    ):
        super().__init__(
            max_batch_size=max_batch_size,
            max_num_pages=max_num_pages,
            page_size=page_size,
            pages=[
                torch.empty(
                    max_num_pages,
                    page_size,
                    config.kv_lora_rank + config.qk_rope_head_dim,
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=True,
                ).to(device)
                for _ in range(config.num_hidden_layers)
            ],
            device=device
        )        
        self.ckv_dim = config.kv_lora_rank
        self.kpe_dim = config.qk_rope_head_dim

    def update(
        self,
        ckv: torch.Tensor,
        kpe: torch.Tensor,
        layer_idx: int,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
    ):
        B = ckv.shape[0]
        seqlen = ckv.shape[1]

        self.append_indptr = torch.arange(
            0, B * seqlen + 1, seqlen, device=self.device
        ).view(B + 1)

        batch_indices, positions = flashinfer.page.get_batch_indices_positions(
            self.append_indptr,
            flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, self.page_size),
            B * seqlen,
        )
        flashinfer.page.append_paged_mla_kv_cache(
            ckv.contiguous().view(-1, self.ckv_dim),
            kpe.contiguous().view(-1, self.kpe_dim),
            batch_indices,
            positions,
            self.pages[layer_idx][..., : self.ckv_dim],
            self.pages[layer_idx][..., self.ckv_dim :],
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
        )

        return self.pages[layer_idx]


class GQAPageTable(PageTable):
    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
        device: str = "cuda",
    ):
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_key_value_heads
        super().__init__(
            max_batch_size=max_batch_size,
            max_num_pages=max_num_pages,
            page_size=page_size,
            pages=[
                torch.empty(
                    max_num_pages,
                    2,
                    page_size,
                    self.num_heads,
                    self.head_dim,
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=True,
                ).to(device)
                for _ in range(config.num_hidden_layers)
            ],
            device=device
        )

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
    ):
        B = key.shape[0]
        seqlen = key.shape[1]

        self.append_indptr = torch.arange(
            0, B * seqlen + 1, seqlen, device=self.device
        ).view(B + 1)

        batch_indices, positions = flashinfer.page.get_batch_indices_positions(
            self.append_indptr,
            flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, self.page_size),
            B * seqlen,
        )
        flashinfer.page.append_paged_kv_cache(
            key.contiguous().view(-1, self.num_heads, self.head_dim),
            value.contiguous().view(-1, self.num_heads, self.head_dim),
            batch_indices,
            positions,
            self.pages[layer_idx],
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_len,
        )

        return self.pages[layer_idx]
