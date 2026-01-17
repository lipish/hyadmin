import ctypes
import os
import re
import sys

import numpy as np
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from heyi._ext.moe import MOE, MOEConfig
from heyi.models.deepseek_v3 import DeepseekV3MoE
from heyi.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from heyi.operators.base import CustomLoadModule
from heyi.operators.cpuinfer import CPUInfer
from heyi.config import Config
from heyi.utils import main_model_registry

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "heyi_ext", "build"))
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "heyi_ext", "build", "Release")
)
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "heyi_ext", "build", "Debug")
)


class KExpertsCPU(nn.Module, CustomLoadModule):
    CPU_INFER = None
    n_obj = 0

    def __init__(
        self,
        config: PretrainedConfig,
        device: str = "cpu",
        out_device: str = "cuda",
    ):
        super().__init__()
        if KExpertsCPU.CPU_INFER is None:
            KExpertsCPU.CPU_INFER = CPUInfer(Config().num_cpu_threads, 1024)
        assert device.lower() == "cpu", "KExpertsCPU can only be loaded on CPU"
        self.config = config
        self.n_routed_experts: int = (
            getattr(config, "n_routed_experts", None) 
            or getattr(config, "num_experts", None)
        )
        self.device = device
        self.out_device = out_device
        self.init_io_buffer()

        self.obj_id = 0

    def load(self, state_dict: dict[str: torch.Tensor], key: str):
        if self.device and self.device.lower() != "cpu":
            raise ValueError("KExpertsCPU can only be loaded on CPU")

        is_fp8 = self.config.weight_dtype == torch.float8_e4m3fn

        for i in range(self.n_routed_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"{key}.{i}.{proj}"
                if f"{base}.weight" not in state_dict or (
                    is_fp8 and f"{base}.weight_scale_inv" not in state_dict
                ):
                    raise KeyError("missing weights")

        # total_start = time.perf_counter()

        # extract_start = time.perf_counter()
        M = self.config.moe_intermediate_size
        H = self.config.hidden_size

        if is_fp8:
            gate = np.empty((self.n_routed_experts, M, H), dtype=np.uint8)
            up = np.empty((self.n_routed_experts, M, H), dtype=np.uint8)
            down = np.empty((self.n_routed_experts, H, M), dtype=np.uint8)
            S_M = M // 128
            S_H = H // 128
            self.gate_inv = np.empty((self.n_routed_experts, S_M, S_H), dtype=np.float32)
            self.up_inv = np.empty((self.n_routed_experts, S_M, S_H), dtype=np.float32)
            self.down_inv = np.empty((self.n_routed_experts, S_H, S_M), dtype=np.float32)
        else:
            gate = np.empty((self.n_routed_experts, M, H), dtype=np.uint16)
            up = np.empty((self.n_routed_experts, M, H), dtype=np.uint16)
            down = np.empty((self.n_routed_experts, H, M), dtype=np.uint16)

        for i in range(self.n_routed_experts):
            for proj, arr, inv_arr in zip(
                ("gate_proj", "up_proj", "down_proj"),
                (gate, up, down),
                (self.gate_inv, self.up_inv, self.down_inv) if is_fp8 else (None, None, None),
            ):
                base = f"{key}.{i}.{proj}"
                if is_fp8:
                    arr[i] = state_dict[f"{base}.weight"].view(torch.uint8).numpy()
                    inv_arr[i] = state_dict[f"{base}.weight_scale_inv"].float().numpy()
                else:
                    arr[i] = state_dict[f"{base}.weight"].view(torch.uint16).numpy()

        # extract_end = time.perf_counter()
        # print(f"Extraction: {extract_end - extract_start}s")

        gate_type = up_type = down_type = 31 if is_fp8 else 30 # 31: FP8; 30: BF16

        def get_ptr(arr):
            return (
                ctypes.addressof(
                    ctypes.cast(
                        arr.ctypes.data, ctypes.POINTER(ctypes.c_uint64)
                    ).contents
                )
                if arr is not None
                else 0
            )

        gate_ptr, up_ptr, down_ptr = map(get_ptr, (gate, up, down))

        if is_fp8:
            gate_inv_ptr, up_inv_ptr, down_inv_ptr = map(
                get_ptr, (self.gate_inv, self.up_inv, self.down_inv)
            )
        else:
            gate_inv_ptr = up_inv_ptr = down_inv_ptr = 0

        moe_config = MOEConfig(
            self.n_routed_experts,              # expert_num
            self.config.num_experts_per_tok,    # routed_expert_num
            self.config.hidden_size,            # hidden_size
            self.config.moe_intermediate_size,  # intermediate_size
            100,                                # group_min_len
            2048,                               # group_max_len
            gate_ptr,                           # gate_proj
            up_ptr,                             # up_proj
            down_ptr,                           # down_proj
            gate_type,                          # gate_type
            up_type,                            # up_type
            down_type,                          # down_type
            30,                                 # hidden_type
            gate_inv_ptr,                       # gate_inv
            up_inv_ptr,                         # up_inv
            down_inv_ptr,                       # down_inv
        )

        # MOE initialization
        # moe_init_start = time.perf_counter()
        self.moe = MOE(moe_config)
        # moe_init_end = time.perf_counter()
        # print(f"MOE initialization time: {moe_init_end - moe_init_start}s")

        del gate, up, down  # remove large buffers
        # gc.collect()

        # total_end = time.perf_counter()
        # print(f"Total load function execution time: {total_end - total_start}s")

        self.cpu_infer = KExpertsCPU.CPU_INFER
        # print(f"{self.obj_id}. KExpertsCPU warmup...")
        self.cpu_infer.submit(self.moe.wrapped_warmup(self.obj_id))
        self.cpu_infer.sync(self.obj_id)
        # print(f"{self.obj_id}. KExpertsCPU warmup done")

        ilayer = int(re.search(r"layers\.(\d*?)\.", key).group(1))
        main_model_registry.expert_modules[ilayer] = self

        if is_fp8:
            self.gate_inv_tensor = torch.tensor(self.gate_inv, dtype=torch.float32)
            self.up_inv_tensor = torch.tensor(self.up_inv, dtype=torch.float32)
            self.down_inv_tensor = torch.tensor(self.down_inv, dtype=torch.float32)

    def init_io_buffer(self):
        maxB = Config().batch_sizes_per_runner[-1]
        self.input_tensor_cpu = torch.zeros(
            (maxB, self.config.hidden_size), device="cpu", pin_memory=True
        )
        self.expert_ids_cpu = torch.zeros(
            (maxB, self.config.num_experts_per_tok),
            device="cpu",
            dtype=torch.long,
            pin_memory=True,
        )
        self.weights_cpu = torch.zeros(
            (maxB, self.config.num_experts_per_tok),
            device="cpu",
            dtype=torch.float32,
            pin_memory=True,
        )
        self.output_cpu = torch.zeros(
            (maxB, self.config.hidden_size),
            device="cpu",
            dtype=torch.bfloat16,
            pin_memory=True,
        )
        self.output_gpu = torch.zeros(
            (maxB, self.config.hidden_size), device=self.out_device
        )

    def submit_for_one_decode(self, input_tensor, expert_ids, weights):
        B = input_tensor.shape[0]
        self.input_tensor_cpu[:B].copy_(input_tensor, non_blocking=True)
        self.expert_ids_cpu[:B].copy_(expert_ids, non_blocking=True)
        self.weights_cpu[:B].copy_(weights, non_blocking=True)
        self.cpu_infer.cuda_launch_host_func(
            torch.cuda.current_stream(self.out_device).cuda_stream,
            self.moe.wrapped_forward(
                self.obj_id,
                B, # qlen
                expert_ids.shape[1], # k
                self.expert_ids_cpu.data_ptr(),
                self.weights_cpu.data_ptr(),
                self.input_tensor_cpu.data_ptr(),
                self.output_cpu.data_ptr(),
            ),
        )

    def sync_for_one_decode(self):
        self.cpu_infer.cuda_launch_host_func(
            torch.cuda.current_stream(self.out_device).cuda_stream,
            self.moe.wrapped_sync(self.obj_id)
        )
        self.output_gpu.copy_(
            self.output_cpu, non_blocking=True
        )
        return self.output_gpu

    def forward(self, input_tensor, expert_ids, weights):
        input_tensor = input_tensor.contiguous().cpu()
        expert_ids = expert_ids.contiguous().cpu()
        weights = weights.contiguous().to(torch.float32).cpu()
        output = torch.empty_like(input_tensor).contiguous()
        self.cpu_infer.submit(
            self.moe.wrapped_forward(
                self.obj_id,
                expert_ids.size(0),
                expert_ids.size(1),
                expert_ids.data_ptr(),
                weights.data_ptr(),
                input_tensor.data_ptr(),
                output.data_ptr(),
            )
        )
        self.cpu_infer.sync(self.obj_id)
        return output.to(device=object.__getattribute__(self, "out_device"))

    def fork(self):
        x = KExpertsCPU(self.config, self.device, self.out_device)
        x.obj_id = KExpertsCPU.n_obj
        KExpertsCPU.n_obj += 1
        x.moe = self.moe
        x.cpu_infer = self.cpu_infer
        return x

    def get_expert_weight(self, iexpert: int):
        gate = torch.empty(
            (self.config.moe_intermediate_size, self.config.hidden_size),
            device="cpu",
            dtype=self.config.weight_dtype,
            pin_memory=True,
        )
        up = torch.empty(
            (self.config.moe_intermediate_size, self.config.hidden_size),
            device="cpu",
            dtype=self.config.weight_dtype,
            pin_memory=True,
        )
        down = torch.empty(
            (self.config.hidden_size, self.config.moe_intermediate_size),
            device="cpu",
            dtype=self.config.weight_dtype,
            pin_memory=True,
        )

        self.cpu_infer.submit(
            self.moe.wrapped_getweight(
                self.obj_id, 
                iexpert, 
                gate.data_ptr(), 
                up.data_ptr(), 
                down.data_ptr()
            )
        )
        self.cpu_infer.sync(self.obj_id)

        is_fp8 = self.config.weight_dtype == torch.float8_e4m3fn
        if is_fp8:
            gate_inv = self.gate_inv_tensor[iexpert]
            up_inv = self.up_inv_tensor[iexpert]
            down_inv = self.down_inv_tensor[iexpert]
        else:
            gate_inv = up_inv = down_inv = None

        return ((gate, gate_inv), (up, up_inv), (down, down_inv))

class KDeepseekV3MoE(DeepseekV3MoE):
    experts: KExpertsCPU

    def forward(self, hidden_states):
        B = hidden_states.shape[0]
        identity = hidden_states
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # only for generate phase
        if (
            sequence_length == 1
            and hasattr(self.experts, "submit_for_one_decode")
            and torch.cuda.is_current_stream_capturing()
        ):
            self.experts.submit_for_one_decode(
                hidden_states, topk_idx, topk_weight
            )
            y_ = self.shared_experts(identity).squeeze(1)
            y = self.experts.sync_for_one_decode()[:B]
            y += y_
            y.resize_(*orig_shape)
            return y

        if self.config.n_shared_experts is not None:
            y_ = self.shared_experts(identity).squeeze(0)

        y = (
            self.moe_kexperts(hidden_states, topk_idx, topk_weight)
            .view(*orig_shape)
            .to(device=hidden_states.device)
        )
        if self.config.n_shared_experts is not None:
            y += y_
        return y

    @torch.no_grad()
    def moe_kexperts(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ) -> torch.Tensor:
        outs = self.experts(x, topk_ids, topk_weight)
        return outs


class KQwen3MoeSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    experts: KExpertsCPU
    def forward(self, hidden_states):
        B = hidden_states.shape[0]
        orig_shape = hidden_states.shape
        sequence_length = orig_shape[1]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits = self.gate(hidden_states)

        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # only for generate phase
        if (
            sequence_length == 1
            and hasattr(self.experts, "submit_for_one_decode")
            and torch.cuda.is_current_stream_capturing()
        ):  # TODO: this branch cause jit bug
            self.experts.submit_for_one_decode(hidden_states, selected_experts, routing_weights)
            y = self.experts.sync_for_one_decode()[:B]
            y.resize_(*orig_shape)
            return y

        y = (
            self.moe_on_cpuinfer(
                hidden_states,
                selected_experts,
                routing_weights,
            )
            .view(*orig_shape)
            .to(device=hidden_states.device)
        )
        return y

    @torch.no_grad()
    def moe_on_cpuinfer(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        outs = torch.empty_like(x)
        outs = self.experts(x, topk_ids, topk_weight)
        return outs


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from transformers import PretrainedConfig

    # === Dummy config ===
    class DummyConfig(PretrainedConfig):
        def __init__(self):
            super().__init__()
            self.n_routed_experts = 4
            self.num_experts_per_tok = 2
            self.hidden_size = 128
            self.moe_intermediate_size = 256

    config = DummyConfig()
    model = KExpertsCPU(config=config)

    # === Fake state_dict ===
    dummy_state_dict = {}
    torch.manual_seed(42)
    for i in range(4):
        dummy_state_dict[f"experts.{i}.gate_proj.weight"] = torch.rand(128, 256).to(torch.float8_e4m3fn)
        dummy_state_dict[f"experts.{i}.gate_proj.weight_scale_inv"] = torch.ones(2).float()
        dummy_state_dict[f"experts.{i}.up_proj.weight"] = torch.rand(128, 256).to(torch.float8_e4m3fn)
        dummy_state_dict[f"experts.{i}.up_proj.weight_scale_inv"] = torch.ones(2).float()
        dummy_state_dict[f"experts.{i}.down_proj.weight"] = torch.rand(256, 128).to(torch.float8_e4m3fn)
        dummy_state_dict[f"experts.{i}.down_proj.weight_scale_inv"] = torch.ones(2).float()

    print("Loading weights...")
    model.load(dummy_state_dict, "experts")

    # === Input data ===
    input_tensor = torch.randn(128, dtype=torch.bfloat16)
    expert_ids = torch.tensor([0, 2], dtype=torch.uint64)
    weights = torch.tensor([0.5, 0.5])

    # === Inference through optimized path ===
    print("Running inference...")
    output = model.forward(input_tensor, expert_ids.unsqueeze(0), weights.unsqueeze(0)).cpu()
    print("Optimized Output:", output.shape, output.dtype)

    # === Reference PyTorch implementation ===
    print("Running PyTorch reference path...")

    torch_output = torch.zeros_like(input_tensor)
    for idx, w in zip(expert_ids.tolist(), weights.tolist()):
        gate_proj = dummy_state_dict[f"experts.{idx}.gate_proj.weight"].bfloat16()
        up_proj = dummy_state_dict[f"experts.{idx}.up_proj.weight"].bfloat16()
        down_proj = dummy_state_dict[f"experts.{idx}.down_proj.weight"].bfloat16()

        # Mocking the computation: gate -> up -> GELU -> down
        x = input_tensor.bfloat16()

        o_gate = F.linear(x, gate_proj.T)
        o_gate = F.silu(o_gate)

        o_up = F.linear(x, up_proj.T)

        x = F.linear(o_gate * o_up, down_proj.T)
        torch_output += 0.5 * x

    torch_output = torch_output.to(dtype=output.dtype).to("cpu")
    print(output)
    print(torch_output)

    # === Comparison ===
    abs_diff = (output - torch_output).abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()
    print(f"\nPrecision comparison against PyTorch:")
    print(f"Max error: {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")

    if max_err < 1e-2:
        print("✅ Precision within expected tolerance.")
    else:
        print("⚠️ High error — may indicate an implementation issue.")
