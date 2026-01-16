import gc
import glob
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers.modeling_utils import load_state_dict

from heyi.operators.base import CustomLoadModule
from heyi.operators.fp8gemm import weight_dequant
from heyi.utils.log import logger


def get_module_from_name(
    module: nn.Module, tensor_name: str
) -> Tuple[str, nn.Module | None, str, bool]:
    def try_get_submodule(module: nn.Module, name: str):
        try:
            return module.get_submodule(submod_name)
        except AttributeError:
            return None

    submod_name = tensor_name
    submodule = module
    strict = True
    if "." in tensor_name:
        submod_name, tensor_name = tensor_name.rsplit(".", 1)
        submodule = try_get_submodule(module, submod_name)
        while not submodule:
            strict = False
            submod_name, _ = submod_name.rsplit(".", 1)
            submodule = try_get_submodule(module, submod_name)

    return submod_name, submodule, tensor_name, strict


def num_sorted(paths: list[str]) -> list[str]:
    """
    Sorts a list of file paths based on the first numeric part in the filename.
    """

    def extract_number(path: str) -> int:
        match = re.search(r"(\d+)", os.path.basename(path))
        return int(match.group(1)) if match else 0x7FFFFFFF

    return sorted(paths, key=extract_number)


class WeightLoader:
    def __init__(
        self,
        model_path: str | os.PathLike,
        gguf_path: Optional[str | os.PathLike] = None,
    ):
        model_path = Path(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Path not found: {model_path}")

        shard_paths = num_sorted(
            glob.glob(os.path.join(model_path, "*.safetensors"), recursive=True)
        )

        self.state_dict = {}
        for path in shard_paths:
            self.state_dict.update(load_state_dict(path))

    @torch.no_grad()
    def load_model(
        self,
        model: nn.Module,
    ):
        start_t = time.perf_counter()
        loaded_custom_load_modules = set()
        prev_ilayer = None
        for key, tensor in self.state_dict.items():
            module_name, module, tensor_name, strict_match = get_module_from_name(
                model, key
            )
            if m := re.match(r"model\.layers\.(\d+)", module_name):
                ilayer = m.group(1)
                if prev_ilayer != ilayer:
                    print(
                        f"\rloading: layer {ilayer}", end="", flush=True
                    )
                    prev_ilayer = ilayer

            if isinstance(module, CustomLoadModule):
                if module in loaded_custom_load_modules:
                    continue
                module.load(self.state_dict, module_name)
                loaded_custom_load_modules.add(module)
            elif strict_match:
                tensor = tensor.cuda()
                if (
                    hasattr(module, "dequantize")
                    and getattr(module, "dequantize")
                    and tensor.dtype == torch.float8_e4m3fn
                ):
                    tensor = weight_dequant(
                        tensor, self.state_dict[key + "_scale_inv"].cuda()
                    ).bfloat16()
                assert module
                module.load_state_dict({tensor_name: tensor}, strict=False, assign=True)
            else:
                (f"Key {key} not found in model. Skipping.")
            # self.state_dict[key] = None
        print()

        gc.collect()
        torch.cuda.empty_cache()
        end_t = time.perf_counter()
        logger.info(f"Weight loading takes {end_t - start_t}s")

        # for module_name, module in model.named_modules():
        #     if isinstance(module, CustomLoadModule) and \
        #         (module not in loaded_custom_load_modules):
        #         module.load()
        #         print(f"supplimentary load for {module_name}")
