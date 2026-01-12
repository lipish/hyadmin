import time
from typing import Any, Callable, List, NamedTuple, Optional, Dict
from uuid import uuid4

import threading


def indstr(d: int, s: str | List):
    if isinstance(s, str):
        return "  " * d + s
    elif isinstance(s, list):
        return "\n".join(list(map(lambda x: "  " * d + x, s)))


Match = NamedTuple("Match", [("len", int), ("node", "PrefixTree.Node")])


class PrefixTree:
    """
    kvcache trie-tree for one batch
    each node is a range of pages
    nodes under same parent don't share any prefix
    (split only happens at the exact difference segment)
    """

    class Node:
        def __init__(
            self,
            page_indices: List[int],
            page_hashs: List[int],
            parent: Optional["PrefixTree.Node"] = None,
            is_root: Optional[bool] = False,
        ):
            self.node_id = uuid4().int if not is_root else 0
            self.page_indices = page_indices
            self.page_hashs = page_hashs
            self.parent = parent
            self._children: Dict[int, "PrefixTree.Node"] = {}
            self.is_root = is_root
            self.timestamp = time.perf_counter()
            self.subtree_size = len(page_hashs)

        @property
        def children(self):
            return self._children.values()

        def _update_subtree_size_bt(self, diff: int):
            """
            update subtree size from self to root
            """
            node = self
            while node:
                node.subtree_size += diff
                node = node.parent

        def _dropch(self, node: "PrefixTree.Node"):
            self._children.pop(node.node_id)
            self._update_subtree_size_bt(-node.subtree_size)

        def _addch(self, node: "PrefixTree.Node"):
            self._children[node.node_id] = node
            self._update_subtree_size_bt(node.subtree_size)

        def _extend_pagelist(self, page_indices: List[int], page_hashs: List[int]):
            assert not self.is_root
            self.page_indices += page_indices
            self.page_hashs += page_hashs
            self._update_subtree_size_bt(len(page_hashs))

        def split(self, position: int):
            """
            split after count: list -> list[:position], list[position:]
            e.g. [a, b, c].split(1) = [a,], [b, c]
            """
            if self.is_root:
                return (self, None)
            assert 0 <= position <= self.len
            if position == 0:
                return (self.parent, self)
            if position == len(self.page_hashs):
                return (self, None)

            assert self.parent
            lnode = PrefixTree.Node(
                self.page_indices[:position],
                self.page_hashs[:position],
                self.parent,
            )
            rnode = PrefixTree.Node(
                self.page_indices[position:],
                self.page_hashs[position:],
                lnode,
            )
            if self.parent:
                self.parent._dropch(self)
                self.parent._addch(lnode)
            lnode._addch(rnode)
            for c in self.children:
                rnode._addch(c)
                c.parent = rnode
            return (lnode, rnode)

        def _update_timestamp_bt(self):
            """
            update timestamp from self to root
            """
            node = self
            while node:
                node.timestamp = time.perf_counter()
                node = node.parent

        def _match(self, page_hashs: List[int]):
            """
            single node match, non-recursive
            """
            min_l = min(len(self.page_hashs), len(page_hashs))
            matched = 0
            for i, (a, b) in enumerate(
                zip(self.page_hashs[:min_l], page_hashs[:min_l])
            ):
                matched = i
                if a != b:
                    break
            else:
                matched = min_l
            return matched

        def treematch(self, page_hashes: List[int]) -> Match:
            all_matched = self.prefix_len
            node = self
            while True:
                matched = node._match(page_hashes)
                page_hashes = page_hashes[matched:]
                all_matched += matched

                if matched != node.len:
                    return Match(all_matched, node)

                for c in node.children:
                    child_matched = c._match(page_hashes)
                    if child_matched > 0:
                        node = c
                        break
                else:
                    return Match(all_matched, node)

        def add(
            self, page_indices: List[int], page_hashs: List[int], at: int
        ) -> "PrefixTree.Node":
            assert len(page_indices) == len(page_hashs)
            # print(f"add {page_indices=}, {page_hashs=}")

            if at == 0:  # no match, add as a new child
                assert self.is_root
                leaf_node = PrefixTree.Node(page_indices, page_hashs, self)
                self._addch(leaf_node)
                return leaf_node

            assert self.parent  # assert node is not root
            lnode, rnode = self.split(at)
            assert lnode
            if rnode is None and not lnode.children:
                lnode._extend_pagelist(
                    page_indices,
                    page_hashs,
                )
                leaf_node = lnode
            else:
                leaf_node = PrefixTree.Node(page_indices, page_hashs, lnode)
                lnode._addch(leaf_node)
            leaf_node._update_timestamp_bt()
            return leaf_node

        def _traverse(
            self,
            *,
            attr: Optional[str] = None,  # reducible
            f: Optional[Callable[["PrefixTree.Node"], Any]] = None,
        ):
            """
            apply `f` or aggregate `attr` in subtree, root-first sequence
            """
            if attr:
                assert f is None, "attr and f cannot be both set"
                f = lambda x: getattr(x, attr)

            assert f is not None, "f must be set"
            ret = f(self)
            for c in self.children:
                ret += c._traverse(f=f)
            return ret

        def prefix_page_indices(self):
            page_indices = []
            node = self
            while node:
                page_indices = node.page_indices + page_indices
                node = node.parent
            return page_indices

        def free(self, npages: int) -> List[int]:
            # print(f"try freeing {npages=} from subtree, root {self.page_hashs}")
            assert npages <= self.subtree_size

            children_size = self.subtree_size - self.len
            if children_size < npages:
                rlen = npages - children_size
                lnode, rnode = self.split(self.len - rlen)
                assert lnode
                assert rnode
                lnode._dropch(rnode)
                # print(f"freeing {rnode.subtree_size}, dropping:\n{rnode}")
                return rnode._traverse(attr="page_indices")

            # Sort children by LRU timestamp (ascending: least recently used first)
            lru_children = sorted(self.children, key=lambda x: x.timestamp)

            # Iterate through sorted children
            freed_pages = []
            for c in lru_children:
                # print(f"child: {c.__repr__()}")
                freed_pages += c.free(min(npages - len(freed_pages), c.subtree_size))
                # print(f"{npages=}, {freed_pages=}, {len(freed_pages)}")
                if len(freed_pages) >= npages:
                    # print("break")
                    break
            return freed_pages

        def __str__(self, d=0):
            s = indstr(
                d,
                [
                    f"+ Node {self.node_id}",
                    f"{self.parent=}",
                    f"{self.page_hashs=}",
                    f"{self.page_indices=}",
                    # f"{self.children=}",
                    f"{self.is_root=}",
                    f"{self.timestamp=}",
                ],
            )
            # indprint(d, f"parent: {self.parent}")
            for c in self.children:
                s += "\n" + c.__str__(d + 1)
            return s

        def __repr__(self) -> str:
            return f"Node({self.node_id}, pages={self.page_hashs})"

        @property
        def len(self) -> int:
            return len(self.page_hashs)

        @property
        def prefix_len(self) -> int:
            return (
                self.len if self.parent == None else self.parent.prefix_len + self.len
            )

    def __init__(self):
        self.root = PrefixTree.Node([], [], None, is_root=True)
        # self.root = CacheTree.Node([], None, is_root=True)
        self.lock = threading.Lock()

    def add(
        self,
        page_indices: List[int],
        page_hashs: list[int],
        matched: Match,
    ) -> Node:
        with self.lock:
            l, node = matched
            at = l - (node.parent.prefix_len if node.parent else 0)
            return node.add(page_indices, page_hashs, at)

    def modify(self, node: "PrefixTree.Node", last_page_hash: int):
        """
        update leaf node's last page hash
        """
        with self.lock:
            assert not node._children
            node.page_hashs[-1] = last_page_hash
            node._update_timestamp_bt()

    def match(self, page_hashs: List[int]):        
        match = self.root.treematch(page_hashs)
        if match.node == self.root:
            return match

        # split at matched point
        with self.lock:
            l, node = match
            at = l - (node.parent.prefix_len if node.parent else 0)
            node, _ = node.split(at)
            assert node
            node._update_timestamp_bt()
            return Match(l, node)

    def free(self, npages: int):
        """
        free at least npages
        """
        with self.lock:
            # print(f"Freeing {npages} pages")
            return self.root.free(npages)

    def __str__(self):
        return "\nPrefixTree:\n" + str(self.root) + "\n"
