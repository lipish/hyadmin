import math
import random
from typing import List, Optional

import torch
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig

from heyi.server.config import Config
from heyi.utils.kvcache.pagetable import PageTable, MLAPageTable, GQAPageTable
from heyi.utils.kvcache.prefixtree import Match, PrefixTree

import triton
import triton.language as tl

def n_pages(n_tokens: int, page_size: int):
    return (n_tokens + page_size - 1) // page_size

@triton.jit
def hash_pages_kernel(
    input_ptr,
    output_ptr,
    num_pages,
    PAGE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_pages:
        return
    base = input_ptr + pid * PAGE_SIZE
    h = 0
    for i in tl.static_range(PAGE_SIZE):
        x = tl.load(base + i)
        h = h * 31 + x
    tl.store(output_ptr + pid, h)


def hash_pages(pages: torch.Tensor) -> torch.Tensor:
    num_pages, page_size = pages.shape
    if not pages.is_contiguous():
        pages = pages.contiguous()
    input_1d = pages.view(-1)
    output = torch.empty(num_pages, dtype=torch.int64, device=pages.device)
    grid = (num_pages,)
    hash_pages_kernel[grid](input_1d, output, num_pages, PAGE_SIZE=page_size)
    return output


def do_page_hash(seq: torch.Tensor, page_size: int, trim: bool=False) -> list:
    """
    Args:
        seq: 1D tensor of token IDs or values
        page_size: number of tokens per page

    Returns:
        List of hashes, one per page (row)
    """
    L = seq.shape[0]
    if trim:
        # Trim the last unfull page
        num_pages = L // page_size
        seq = seq[:num_pages * page_size]
    else:
        # Pad with zeros at the end (or any sentinel if needed)
        num_pages = math.ceil(L / page_size)
        padded_len = num_pages * page_size

        if L < padded_len:
            pad_len = padded_len - L
            seq = torch.cat(
                [seq, torch.full((pad_len,), -1, dtype=seq.dtype, device=seq.device)]
            )
        
    # Reshape to [num_pages, page_size]
    pages = seq.reshape(num_pages, page_size).int()
    page_hashes = hash_pages(pages)
    page_hashes = page_hashes.tolist()

    # Compute hashes per row

    # page_hashes = []
    # for row in pages:
    #     ans = ""
    #     for i in row.tolist():
    #         ans += str(i) + ", "
    #     page_hashes.append(ans)

    return page_hashes


class PagedKVCache(Cache):
    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
        device: int = 0,
        prefix_tree: Optional[PrefixTree] = None,
        page_table: Optional[PageTable] = None,
    ):
        super().__init__()
        self.config = config
        self.max_batch_size = max_batch_size
        self.max_num_pages = max_num_pages
        self.page_size = page_size
        self.device = device

        if prefix_tree:
            self.prefix_tree = prefix_tree
        else:
            self.prefix_tree = PrefixTree()

        self.page_table = page_table

        self.buffers = {
            "page_indices": torch.zeros(
                max_num_pages, dtype=torch.int32, device=self.device
            ),
            "page_indptr": torch.zeros(
                max_batch_size + 1, dtype=torch.int32, device=self.device
            ),
            "last_page_len": torch.zeros(
                max_batch_size, dtype=torch.int32, device=self.device
            ),
        }

    def fork(self):
        """
        Fork the cache for a new runner.
        """
        x = self.__class__(
            self.config,
            self.max_batch_size,
            self.max_num_pages,
            self.page_size,
            self.device,
            self.prefix_tree,
            self.page_table,
        )
        return x

    def match(self, all_ids: torch.Tensor):
        batch_size = all_ids.shape[0]
        matches: List[Match] = []
        for seq in all_ids:
            page_hashs = do_page_hash(seq, self.page_size, trim=True)
            # print(f"Matching seq: {seq} -> page_hashs: {page_hashs}")
            matches.append(self.prefix_tree.match(page_hashs))
        return matches

    def plan(
        self, matches: List[Match], all_ids: List[torch.Tensor], return_matches: bool = True
    ):
        B = len(matches)
        self.B = B
        assert len(all_ids) == B, "kv_append_length must match batch size"

        page_indices = self.buffers["page_indices"]
        page_indptr = self.buffers["page_indptr"]
        last_page_len = self.buffers["last_page_len"]

        if return_matches:
            ret_matches = []

        for i, match in enumerate(matches):
            (l, node) = match

            if node == self.prefix_tree.root:
                append_page_ids = all_ids[i]
                pages_needed = (
                    append_page_ids.shape[0] + self.page_size - 1
                ) // self.page_size
            else:
                """
                Example:
                    previous sequence:  | 0 1 2 | 3 4   |
                    all_ids:            | 0 1 2 | 3 4 5 | 6 7 8 | 9     |
                    relative_ids:               | 3 4 5 | 6 7 8 | 9     |
                    last_page_ids:              | 3 4 5 |
                    append_page_ids:                    | 6 7 8 | 9     |

                    previous sequence:  | 0 1 2 | 3     |
                    all_ids:            | 0 1 2 | 3 4   |
                    relative_ids:               | 3 4   |
                    last_page_ids:              | 3 4   |
                    append_page_ids:                    |

                    previous sequence:  | 0 1 2 | 3 4 5 |       |
                    all_ids:            | 0 1 2 | 3 4 5 | 6 7   |
                    relative_ids:                       | 6 7   |
                    last_page_ids:                      | 6 7   |
                    append_page_ids:                            |
                """

                relative_ids = all_ids[i][(l - 1) * self.page_size :]

                last_page_ids = relative_ids[: self.page_size]
                append_page_ids = relative_ids[self.page_size :]

                if (
                    l == node.prefix_len
                    and self.page_table.page_filled_len[node.page_indices[-1]]
                    < self.page_size
                ):
                    self.prefix_tree.modify(
                        node, do_page_hash(last_page_ids, self.page_size)[0]
                    )

                pages_needed = (
                    append_page_ids.shape[0] + self.page_size - 1
                ) // self.page_size
                if pages_needed > 0:
                    self.page_table.set_page_filled_len(
                        torch.tensor(
                            [node.page_indices[l - node.prefix_len - 1]],
                            device=self.device,
                        ),
                        torch.tensor([self.page_size], device=self.device, dtype=torch.int),
                    )

            if pages_needed:
                if pages_needed > self.page_table.n_free_pages:
                    # print(
                    #     f"Not enough free pages: {pages_needed=}, {self.page_table.n_free_pages=}"
                    # )
                    # print(
                    #     f"Requesting {pages_needed - self.page_table.n_free_pages} more pages"
                    # )
                    self.page_table.free(
                        self.prefix_tree.free(
                            pages_needed - self.page_table.n_free_pages
                        )
                    )

                new_pages = self.page_table.allocate(pages_needed)
                last_page = new_pages[-1]

                page_filled_len = [self.page_size for _ in new_pages[:-1]] + [
                    append_page_ids.shape[0] % self.page_size or self.page_size
                ]
                self.page_table.set_page_filled_len(
                    torch.tensor(new_pages, device=self.device, dtype=torch.int),
                    torch.tensor(page_filled_len, device=self.device, dtype=torch.int),
                )

                leaf_node: PrefixTree.Node = self.prefix_tree.add(
                    new_pages, do_page_hash(append_page_ids, self.page_size), match
                )
            else:
                assert node.parent
                leaf_node, _ = node.split(l - node.parent.prefix_len)
                assert leaf_node
                last_page = leaf_node.page_indices[-1]

                self.page_table.set_page_filled_len(
                    torch.tensor([last_page], device=self.device),
                    torch.tensor([last_page_ids.shape[0]], device=self.device, dtype=torch.int),
                )

            page_indptr[i + 1] = page_indptr[i] + leaf_node.prefix_len
            page_indices[page_indptr[i] : page_indptr[i + 1]] = torch.tensor(
                leaf_node.prefix_page_indices(), dtype=torch.int
            )
            last_page_len[i] = self.page_table.page_filled_len[last_page]

            if return_matches:
                ret_matches.append(Match(leaf_node.prefix_len, leaf_node))

        if return_matches:
            return ret_matches

    def get_seq_length(self, layer_idx: Optional[int] = 0):
        return (
            self.page_size * (torch.diff(self.buffers["page_indptr"]) - 1)
            + self.buffers["last_page_len"]
        )[: self.B]

    def get_max_cache_shape(self) -> int | None:
        return None
    
    def to_(self, device, layer: Optional[int]=None):
        return self.page_table.to_(device, layer)

    def __str__(self):
        page_indices = self.buffers["page_indices"]
        page_indptr = self.buffers["page_indptr"]
        last_page_len = self.buffers["last_page_len"]
        return (
            f"PagedMLACache:\n"
            f"{page_indices=}\n"
            f"{page_indptr=}\n"
            f"{last_page_len=})\n"
        )

class PagedMLACache(PagedKVCache):
    page_table: MLAPageTable
    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
        device: str = "cuda",
        prefix_tree: Optional[PrefixTree] = None,
        page_table: Optional[PageTable] = None,
    ):
        if page_table is None:
            page_table = MLAPageTable(
                config, max_batch_size, max_num_pages, page_size, device
            )
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_num_pages=max_num_pages,
            page_size=page_size,
            device=device,
            prefix_tree=prefix_tree,
            page_table = page_table,
        )


    def update(
        self,
        ckv: torch.Tensor,
        kpe: torch.Tensor,
        layer_idx: int,
    ):
        page_table = self.page_table.update(
            ckv,
            kpe,
            layer_idx,
            self.buffers["page_indices"][:],
            self.buffers["page_indptr"][: self.B + 1],
            self.buffers["last_page_len"][: self.B],
        )
        return page_table
    
class PagedGQACache(PagedKVCache):
    page_table: GQAPageTable
    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_num_pages: int,
        page_size: int,
        device: str = "cuda",
        prefix_tree: Optional[PrefixTree] = None,
        page_table: Optional[PageTable] = None,
    ):
        if page_table is None:
            page_table = GQAPageTable(
                config, max_batch_size, max_num_pages, page_size, device
            )
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_num_pages=max_num_pages,
            page_size=page_size,
            device=device,
            prefix_tree=prefix_tree,
            page_table = page_table,
        )
        self.buffers["qo_indptr"] = torch.empty(max_batch_size, dtype=torch.int32, device="cuda")

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ):
        page_table = self.page_table.update(
            key,
            value,
            layer_idx,
            self.buffers["page_indices"][:],
            self.buffers["page_indptr"][: self.B + 1],
            self.buffers["last_page_len"][: self.B],
        )
        return page_table
    

if __name__ == "__main__":
    # Example usage
    torch.set_default_device("cuda:0")
    config = PretrainedConfig(
        num_hidden_layers=1, kv_lora_rank=512, qk_rope_head_dim=64
    )
    PAGE_SIZE = 64
    cache = PagedMLACache(
        config, max_batch_size=4, max_num_pages=2000, page_size=PAGE_SIZE
    )

    N = 10000
    for test in range(N):

        B = random.randint(1, 2)
        print("\n")
        print("-" * 10, f"TEST [{test+1}/{N}]", "-" * 10)
        print("\n")
        print(f"Batch size: {B}")

        prompt_len = random.randint(1, 30000)
        decode_len = 4000

        seqlen = prompt_len + decode_len
        all_ids = torch.randint(0, 2, (B, seqlen), dtype=torch.int32)
        p_ids = all_ids[..., :prompt_len]
        d_ids = all_ids[..., prompt_len:]

        # print(f"{prompt_len=}, {decode_len=}, {all_ids=}")

        ckv = torch.randn(B, seqlen, 512).bfloat16()
        kpe = torch.randn(B, seqlen, 64).bfloat16()
        p_ckv = ckv[:, :prompt_len]
        d_ckv = ckv[:, prompt_len:]
        p_kpe = kpe[:, :prompt_len]
        d_kpe = kpe[:, prompt_len:]

        print("[1/5] MATCH")
        matches = cache.match(p_ids)

        print("[2/5] PLAN")
        matches = cache.plan(matches, p_ids, return_matches=True)
        # print(cache)
        # print(cache.prefix_tree.root)

        print("[3/5] UPDATE")
        pagetable = cache.update(p_ckv, p_kpe, layer_idx=0)

        print("[4/5] CHECK PREFILL DONE")
        try:
            assert matches
            for i, (l, node) in enumerate(matches):
                assert l == len(do_page_hash(p_ids[i], PAGE_SIZE))
        except Exception as e:
            print("[ERROR] ON PREFILL DONE LEN CHECK")
            print(e)
            import IPython; IPython.embed()

        try:
            assert matches
            for i, (l, node) in enumerate(matches):
                ckv_kpe_cached = pagetable[node.prefix_page_indices()].view(-1, 576)[: prompt_len]
                ckv_kpe_actual = torch.concat((p_ckv[i], p_kpe[i]), dim=-1)
                assert (ckv_kpe_cached == ckv_kpe_actual).all()
        except Exception as e:
            print("[ERROR] ON PREFILL DONE VAL CHECK")
            print(e)
            import IPython; IPython.embed()

        cur_seqlen = prompt_len
        print("[5/5] DECODE & CHECK")
        for i in range(decode_len):
            cur_seqlen += 1
            matches = cache.plan(
                matches, all_ids[..., :cur_seqlen], return_matches=True
            )
            pagetable = cache.update(d_ckv[:, i : i + 1], d_kpe[:, i : i + 1], 0)
            # print(f"----- STATE BEGIN -----")
            # print(f"after {i}")
            # print(cache)
            # print(cache.page_table)
            # print(cache.prefix_tree)
            # print(f"------ STATE END ------")


        try:
            assert matches
            for i, (l, node) in enumerate(matches):
                assert l == len(do_page_hash(all_ids[i], PAGE_SIZE))
        except Exception as e:
            print("[ERROR] ON DECODE DONE; LEN CHECK")
            print(e)
            import IPython; IPython.embed()


        try:
            assert matches
            for i, (l, node) in enumerate(matches):
                ckv_kpe_cached = pagetable[node.prefix_page_indices()].view(-1, 576)[: seqlen]
                ckv_kpe_actual = torch.concat((ckv[i], kpe[i]), dim=-1)
                assert (ckv_kpe_cached == ckv_kpe_actual).all()
        except Exception as e:
            print("[ERROR] ON DECODE DONE; VAL CHECK")
            print(e)
            import IPython; IPython.embed()
