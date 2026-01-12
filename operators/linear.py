#!/usr/bin/env python
# coding=utf-8
"""
Description  :
Author       : Azure-Tang, Boxin Zhang
Date         : 2024-07-25 11:25:24
Version      : 0.1.0
LastEditors  : Azure
LastEditTime : 2024-08-29 09:11:16
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
"""


import os
import sys

import torch
from torch import nn

from heyi.operators.base import CustomLoadModule
from heyi.operators.fp8gemm import act_quant, fp8_gemm
from heyi.utils import main_model_registry

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "heyi_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "heyi_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "heyi_ext", "build", "Debug"))
from typing import Dict


class KLinearBase(nn.Module, CustomLoadModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: str = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.device = device

    def load_weight(self, key: str, state_dict: Dict[str, torch.Tensor]):
        device = self.device

        # using safetensor_loader
        tensor = state_dict[key + ".weight"]
        if "_proj" not in key:
            return nn.Parameter(tensor)
        if (weight_scale_inv := state_dict.get(key + ".weight_scale_inv")) is not None:
            return nn.Parameter(tensor), nn.Parameter(weight_scale_inv)
        else:
            return nn.Parameter(tensor)

class KLinearTorch(KLinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: str = "cuda",
    ):
        super().__init__(in_features, out_features, bias, device)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.device = device


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        out_device = x.device
        x = x.to(device=self.device, dtype=self.dtype)
        x = x @ self.weight
        if self.has_bias:
            x = x + self.bias
        x = x.to(dtype=dtype, device=out_device)
        return x

    def load(self, state_dict: dict[str, torch.Tensor], key: str):
        device = self.device
        w = self.load_weight(key, state_dict)
        # else: self.out_features = w.shape[0], self.in_features = w.shape[1]
        
        if isinstance(w, nn.Parameter):
            try:
                self.weight = w.to(dtype=self.dtype).view(self.out_features, self.in_features).T
            except: 
                self.weight = w.to(dtype=self.dtype).T
            self.has_bias = False
        elif isinstance(w, tuple):
            try:
                self.weight = w[0].to(dtype=self.dtype).view(self.out_features, self.in_features).T
            except:
                self.weight = w[0].to(dtype=self.dtype).T
            self.bias = w[1].to(dtype=self.dtype)
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        # self.linear = self.linear.to(device)
        self.weight = self.weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
        self.loaded = True
        main_model_registry.linear_modules[key] = self

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.has_bias:
            self.bias = None

    def to_(self, device: str):
        if self.weight is not None:
            self.weight = self.weight.to(device=device, non_blocking=True)
        if self.has_bias:
            self.bias = self.bias.to(device=device, non_blocking=True)

    def fork(self):
        x = KLinearTorch(
            self.in_features, 
            self.out_features,
            self.bias,
            self.device,
        )
        x.weight = self.weight
        if hasattr(self, "weight_scale_inv"):
            x.weight_scale_inv = self.weight_scale_inv 
        if hasattr(self, "bias"):
            x.bias = self.bias
        return x


class KLinearFP8(KLinearBase):
    has_bias: bool
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device: str = "cuda",
        block_size: int = 128,
    ):
        super().__init__(in_features, out_features, bias, device)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.block_size = block_size
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        orig_dtype = x.dtype
        x_quantized, scale_x = act_quant(x, self.block_size)
        y = fp8_gemm(x_quantized, scale_x, self.weight, self.weight_scale_inv)
        return y.to(dtype=orig_dtype)

    def load(self, state_dict: dict[str, torch.Tensor], key: str):
        device = self.device
        w = self.load_weight(key, state_dict)
        ### TODO fit weight_inv format
        if isinstance(w, tuple):
            self.weight = w[0].to(device)
            self.weight_scale_inv = w[1].to(device, dtype=torch.float32)
            self.has_bias = False
        else:
            raise ValueError("Invalid weight type")
        self.weight = self.weight.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)
        main_model_registry.linear_modules[key] = self

    def unload(self):
        if self.weight is not None:
            self.weight = None
        if self.has_bias:
            self.bias = None

    def to_(self, device: str):
        if self.weight is not None:
            self.weight = self.weight.to(device=device, non_blocking=True)
        if self.has_bias:
            self.bias = self.bias.to(device=device, non_blocking=True)

    def fork(self):
        x = KLinearFP8(
            self.in_features, 
            self.out_features,
            self.bias,
            self.device,
            self.block_size
        )
        x.weight = self.weight
        if hasattr(self, "weight_scale_inv"):
            x.weight_scale_inv = self.weight_scale_inv 
        if hasattr(self, "bias"):
            x.bias = self.bias
        return x
        


LINEAR_MAP: Dict[str, KLinearBase] = {
    "KLinearFP8": KLinearFP8,
    "KLinearTorch": KLinearTorch,
}


def KLinear(
    linear: nn.Linear = None,
    in_features: int = None,
    out_features: int = None,
    bias: bool = False,
    op: str = "KLinearFP8",
) -> KLinearFP8:
    """
    Factory function for heyiLinear modules.
    Args:
        op (str, optional): The type of linear operator to use. Defaults to "KLinearTorch".
    """
    if not linear and (not in_features or not out_features):
        raise ValueError("Either linear or in_features and out_features must be provided.")
    if linear:
        in_features, out_features, bias = linear.in_features, linear.out_features, linear.bias
    return LINEAR_MAP[op](in_features, out_features, bias)
