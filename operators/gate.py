import copy
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from heyi.models.deepseek_v3 import MoEGate
from heyi.operators.base import CustomLoadModule
from heyi.operators.linear import KLinear
from heyi.utils.weight_loader import WeightLoader


class KMoEGateBase(MoEGate, CustomLoadModule):
    def __init__(self, config: PretrainedConfig, device: str="cuda"):
        super().__init__(config)
        self.device = device

    def load_weights(self, key: str, gguf_loader: WeightLoader):
        device = self.device
        key = ".".join(key.split(".")[:-1])
        if gguf_loader.safetensor_loader is not None:
            targets = [".gate.weight", ".gate.e_score_correction_bias"]
            weight = gguf_loader.safetensor_loader.load_tensor(key + ".gate.weight") 
            e_score_correction_bias = gguf_loader.safetensor_loader.load_tensor(key + ".gate.e_score_correction_bias")
            weight_type = weight.dtype
            e_score_correction_bias_type = e_score_correction_bias.dtype
            res = {"weight": weight, "e_score_correction_bias": e_score_correction_bias,  "weight_type": weight_type, "e_score_correction_bias_type": e_score_correction_bias_type}
        elif key + ".ffn_gate_inp.weight" in gguf_loader.tensor_info:
            targets = [".ffn_gate_inp.weight", ".exp_probs_b.bias"]
            tensors = self.load_multi(key, targets, device=device, gguf_loader=gguf_loader)
            weight = tensors[".ffn_gate_inp.weight"]
            e_score_correction_bias = tensors[".exp_probs_b.bias"]
            weight_type = gguf_loader.tensor_info[key + ".ffn_gate_inp.weight"]["ggml_type"]
            e_score_correction_bias_type = gguf_loader.tensor_info[key + ".exp_probs_b.bias"]["ggml_type"]
        else:
            raise ValueError(f"Experts {key} not found in gguf_loader")
        res = {"weight": weight, "e_score_correction_bias": e_score_correction_bias,  "weight_type": weight_type, "e_score_correction_bias_type": e_score_correction_bias_type}
        return res
    
    def load_multi(self, key: str, keys: list[str], device: str = "cpu", gguf_loader: WeightLoader=None):
        tensors = {}
        for k in keys:
            tensors[k] = gguf_loader.load_gguf_tensor(key + k, device=device)
        return tensors


class KMoEGate(KMoEGateBase):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

    def load(self, key: str, gguf_loader: WeightLoader):
        device = self.device
        w = self.load_weights(key, gguf_loader)
        
        if isinstance(w, dict):
            self.weight_type = w["weight_type"]
            self.e_score_correction_bias_type = w["e_score_correction_bias_type"]
            self.weight = nn.Parameter(w["weight"])
            self.e_score_correction_bias = nn.Parameter(w["e_score_correction_bias"])
        else:
            raise ValueError("Invalid weight type")
        self.weight = nn.Parameter(self.weight.to(device))
        self.e_score_correction_bias = nn.Parameter(self.e_score_correction_bias.to(device))

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.e_score_correction_bias is not None:
            self.e_score_correction_bias = None

    def fork(self):
        x = copy.deepcopy(self)
        if self.weight is not None:
            x.weight = self.weight
        if self.e_score_correction_bias is not None:
            x.e_score_correction_bias = self.e_score_correction_bias