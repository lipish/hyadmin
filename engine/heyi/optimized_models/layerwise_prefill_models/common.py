import re
import time
from typing import Optional, Tuple

import torch
from torch import nn

from heyi.models.qwen3_moe import Qwen3MoeMLP
from heyi.operators.linear import KLinearFP8, KLinearTorch
from heyi.utils import main_model_registry
from heyi.utils.ring_buffer import RingBufferMgr


class LP_LinearTorch_FromGPU(KLinearTorch):
    def __init__(self, linear: nn.Linear):
        in_features, out_features, bias = (
            linear.in_features,
            linear.out_features,
            linear.bias,
        )
        super().__init__(in_features, out_features, bias)

    def load(self, state_dict: dict[str, torch.Tensor], key: str):
        """dummy load: record state_dict and key, real load happens in `on()`"""
        # self.state_dict = state_dict
        self.key = key

    def on(self):
        self.weight = main_model_registry.linear_modules[self.key].weight.to(
            self.device, non_blocking=True
        )

    def off(self):
        del self.weight


class LP_LinearTorch_FromCPUWithBuffer(LP_LinearTorch_FromGPU):
    def __init__(
        self,
        linear: nn.Linear,
        parent_expert: Optional["LP_MLP_FromCPUWithBuffer"] = None,
    ):
        super().__init__(linear)
        if parent_expert:
            self.parent_expert_register = parent_expert.register
        else:
            self.parent_expert_register = None

    def load(self, state_dict: dict[str, torch.Tensor], key: str):
        self.key = key
        if self.parent_expert_register:
            self.parent_expert_register(self.key)
        pass

    def on(self, buffer: torch.Tensor, src: torch.Tensor):
        w, _ = src
        self.weight = buffer.T
        buffer.copy_(w, non_blocking=True)


class LP_LinearFP8_FromGPU(KLinearFP8):
    def __init__(self, linear: nn.Linear):
        in_features, out_features, bias = (
            linear.in_features,
            linear.out_features,
            linear.bias,
        )
        super().__init__(in_features, out_features, bias)

    def load(self, state_dict: dict[str, torch.Tensor], key: str):
        """dummy load: record state_dict and key, real load happens in `on()`"""
        # self.state_dict = state_dict
        self.key = key

    def on(self):
        self.weight = main_model_registry.linear_modules[self.key].weight.to(
            self.device, non_blocking=True
        )
        self.weight_scale_inv = main_model_registry.linear_modules[
            self.key
        ].weight_scale_inv.to(self.device, non_blocking=True)

    def off(self):
        del self.weight
        del self.weight_scale_inv


class LP_LinearFP8_FromCPUWithBuffer(LP_LinearFP8_FromGPU):
    def __init__(
        self,
        linear: nn.Linear,
        parent_expert: Optional["LP_MLP_FromCPUWithBuffer"] = None,
    ):
        super().__init__(linear)
        if parent_expert:
            self.parent_expert_register = parent_expert.register
        else:
            self.parent_expert_register = None

    def load(self, state_dict: dict[str, torch.Tensor], key: str):
        self.key = key
        if self.parent_expert_register:
            self.parent_expert_register(self.key)
        pass

    def on(self, buffer: torch.Tensor, src: Tuple[torch.Tensor, torch.Tensor]):
        w, w_inv = src
        self.weight_scale_inv = w_inv.to(self.device)
        self.weight = buffer
        buffer.copy_(w, non_blocking=True)


class LP_MLP_FromGPU(Qwen3MoeMLP):  # same to deepseek_v3's mlp
    def __init__(self, config, intermediate_size=None):
        super().__init__(config, intermediate_size)
        if config.weight_dtype == torch.float8_e4m3fn:
            Linear = LP_LinearFP8_FromGPU
        else:
            Linear = LP_LinearTorch_FromGPU
        self.gate_proj = Linear(self.gate_proj)
        self.up_proj = Linear(self.up_proj)
        self.down_proj = Linear(self.down_proj)

    def on(self):
        self.gate_proj.on()
        self.up_proj.on()
        self.down_proj.on()

    def off(self):
        self.gate_proj.off()
        self.up_proj.off()
        self.down_proj.off()

    def forward(self, x):
        CHUNK_SIZE = 32768
        for l in range(0, x.shape[1], CHUNK_SIZE):
            r = min(l + CHUNK_SIZE, x.shape[1])
            x[:, l:r] = self.down_proj(
                self.act_fn(self.gate_proj(x[:, l:r])) * self.up_proj(x[:, l:r])
            )
        # x = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return x


class LP_MLP_FromCPUWithBuffer(LP_MLP_FromGPU):

    def __init__(
        self,
        config,
        ring_buffer_mgr: RingBufferMgr,
        intermediate_size=None,
    ):
        super().__init__(config, intermediate_size)
        if config.weight_dtype == torch.float8_e4m3fn:
            Linear = LP_LinearFP8_FromCPUWithBuffer
        else:
            Linear = LP_LinearTorch_FromCPUWithBuffer
        self.gate_proj = Linear(self.gate_proj, self)
        self.up_proj = Linear(self.up_proj)
        self.down_proj = Linear(self.down_proj)
        self.ring_buffer_mgr = ring_buffer_mgr

    def register(self, key: str):
        self.ilayer = int(re.search(r"layers\.(\d*?)\.", key).group(1))
        self.iexpert = int(re.search(r"experts\.(\d*?)\.", key).group(1))
        # print(f"LAYER[{self.ilayer}]; EXP[{self.iexpert}]")

    def on(self):

        # torch.cuda.nvtx.range_push(f"[MOE W/ BUF] NO FREE SLOT")
        while not self.ring_buffer_mgr.slot_available():
            time.sleep(0.01)
        # torch.cuda.nvtx.range_pop()

        expert_from_cpu = main_model_registry.expert_modules[
            self.ilayer
        ].get_expert_weight(self.iexpert)

        with self.ring_buffer_mgr.ring_buffer_lock:
            ptr = self.ring_buffer_mgr.ptr
            buffers = self.ring_buffer_mgr.buffers
            self.gate_proj.on(buffers["gate"][ptr], expert_from_cpu[0])
            self.up_proj.on(buffers["up"][ptr], expert_from_cpu[1])
            self.down_proj.on(buffers["down"][ptr], expert_from_cpu[2])
            self.ring_buffer_mgr.on_push()

    def off(self):
        super().off()
        self.ring_buffer_mgr.pop()
