import torch

def deepcopy_to_cpu(obj):
    """Deep-copy args/kwargs, putting all tensors on CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(deepcopy_to_cpu(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: deepcopy_to_cpu(v) for k, v in obj.items()}
    else:
        return obj

def move_to_device(obj, device: str):
    """Recursively move all tensors in a nested structure to the given device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj
