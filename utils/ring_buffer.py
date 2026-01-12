import torch
import threading
from typing import Dict


class RingBufferMgr:
    """single-producer, single-consumer ring buffer"""

    ring_buffer_lock = threading.RLock()

    def reset(self, buffers: Dict[str, torch.Tensor], length: int):
        with self.ring_buffer_lock:
            self.length = length
            self.buffers = buffers
            self.ptr = 0
            self.n_used = 0

    def to_(self, device):
        # print(f"[RING BUF MGR]: MOVE RING BUFFERS TO {device}")
        for k in self.buffers:
            self.buffers[k] = self.buffers[k].to(device, non_blocking=True)

    def pop(self):
        with self.ring_buffer_lock:
            self.n_used -= 1
            assert 0 <= self.n_used <= self.length

    def on_push(self):
        self.ptr = (self.ptr + 1) % self.length
        self.n_used += 1
        assert 0 <= self.n_used <= self.length

    def slot_available(self):
        return self.n_used < self.length
