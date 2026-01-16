import torch

from heyi.operators.base import CustomLoadModule

def fork_model(source: torch.nn.Module, target: torch.nn.Module):
    """
    Create a forked copy of a PyTorch model where:
    - Constant weights/buffers are shared with the original model (memory efficient)
    - Runtime buffers (input/output buffers etc.) are newly allocated
    - Custom modules are properly forked using their fork() method

    Args:
        source: Original model to fork from
        target: Target model that will receive the forked components

    Returns:
        The target model with components forked from source
    """
    # print("@@@@@ fork torch params @@@@@")
    for name, param in source.named_parameters():
        module_name, param_name = name.rsplit('.', 1)
        target_module = dict(target.named_modules())[module_name]
        setattr(target_module, param_name, param)
        # print(f"linked {name}: {param_name}")

    # print("@@@@@ fork torch buffers @@@@@")
    for name, buffer in source.named_buffers():
        module_name, buffer_name = name.rsplit('.', 1)
        target_module = dict(target.named_modules())[module_name]
        setattr(target_module, buffer_name, buffer)
        # print(f"linked {name}")

    # print("@@@@@ fork custom modules @@@@@")
    for name, module in source.named_modules():
        if isinstance(module, CustomLoadModule):
            if "." in name:
                module_prefix, module_name = name.rsplit('.', 1)
                target_module = dict(target.named_modules())[module_prefix]
                setattr(target_module, module_name, module.fork())
                # print(f"forked {name}")
            else:
                setattr(target, name, module.fork())
    return target


