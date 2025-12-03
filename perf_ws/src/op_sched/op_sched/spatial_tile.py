import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_halo(module):
    """
    Automatically calculate the required halo size for a module based on its layers.
    
    The halo should be large enough to cover the receptive field of all operations
    in the module to ensure correct boundary computations.
    
    Args:
        module (nn.Module): The module to analyze
        
    Returns:
        int: Recommended halo size in pixels (minimum 0 for modules without spatial operations)
        
    Examples:
        >>> # Module without spatial operations
        >>> bn = nn.BatchNorm2d(16)
        >>> halo = calculate_halo(bn)
        >>> print(halo)  # 0
        
        >>> # Simple convolution
        >>> conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        >>> halo = calculate_halo(conv)
        >>> print(halo)  # 1
        
        >>> # Dilated convolution
        >>> conv = nn.Conv2d(16, 32, kernel_size=5, dilation=2, padding=4)
        >>> halo = calculate_halo(conv)
        >>> print(halo)  # 4
    """
    max_halo = 0
    
    # Recursively examine all submodules
    for name, submodule in module.named_modules():
        halo = 0
        
        # Convolutional layers
        if isinstance(submodule, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            kernel_size = submodule.kernel_size
            dilation = submodule.dilation
            
            # Handle tuple parameters (for 2D/3D convs)
            if isinstance(kernel_size, tuple):
                kernel_size = max(kernel_size)
            if isinstance(dilation, tuple):
                dilation = max(dilation)
            
            # Calculate receptive field contribution
            # halo = dilation * (kernel_size // 2)
            halo = dilation * (kernel_size // 2)
        
        # Pooling layers
        elif isinstance(submodule, (nn.MaxPool2d, nn.AvgPool2d, nn.MaxPool1d, nn.AvgPool1d)):
            kernel_size = submodule.kernel_size
            
            if isinstance(kernel_size, tuple):
                kernel_size = max(kernel_size)
            
            halo = kernel_size // 2
        
        # Dilated convolutions or atrous spatial pyramid pooling
        elif isinstance(submodule, nn.ConvTranspose2d):
            kernel_size = submodule.kernel_size
            
            if isinstance(kernel_size, tuple):
                kernel_size = max(kernel_size)
            
            halo = kernel_size // 2
        
        max_halo = max(max_halo, halo)
    
    # For sequential/nested modules, we need to account for accumulation
    # Check if it's a Sequential or contains multiple conv layers
    conv_count = sum(1 for m in module.modules() 
                     if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d)))
    
    if conv_count > 1:
        # For stacked convolutions, add extra padding
        # This is a conservative estimate
        max_halo = int(max_halo * math.sqrt(conv_count))
    
    return max(max_halo, 0)  # Minimum halo of 0 (for modules without spatial operations)


class SpatiallyTiledModule(nn.Module):
    """
    Wrap an nn.Module so its input is processed in HxW tiles.

    Assumes:
        - input:  N x C x H x W
        - output: N x C_out x H x W  (same spatial size)
    
    Args:
        module (nn.Module): The module to wrap
        splits_h (int): Number of splits along height dimension
        splits_w (int): Number of splits along width dimension
        halo (int, optional): Pixels of overlap on each tile boundary.
                             If None, automatically calculated based on module's layers.
    """

    def __init__(self, module: nn.Module,
                 splits_h: int = 1,
                 splits_w: int = 1):
        super().__init__()
        self.module = module
        self.splits_h = splits_h
        self.splits_w = splits_w

        self.halo = calculate_halo(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        if self.splits_h == 1 and self.splits_w == 1:
            # No partitioning – just run the module
            return self.module(x)

        # Compute tile sizes (allow non-divisible H/W, last tile may be bigger)
        h_splits = self._compute_splits(H, self.splits_h)
        w_splits = self._compute_splits(W, self.splits_w)

        tiles_out = []
        h_start = 0
        for h_size in h_splits:
            w_row_out = []
            w_start = 0
            for w_size in w_splits:
                h_end = h_start + h_size
                w_end = w_start + w_size

                # Expand region with halo (clamped to [0, H/W])
                h0 = max(0, h_start - self.halo)
                h1 = min(H, h_end + self.halo)
                w0 = max(0, w_start - self.halo)
                w1 = min(W, w_end + self.halo)

                x_tile = x[:, :, h0:h1, w0:w1]  # N x C x H_tile_ext x W_tile_ext

                # Run the wrapped module on this tile
                y_tile = self.module(x_tile)    # N x C_out x H_tile_ext x W_tile_ext

                # Now crop away extra halo from the output
                # (adjust crop indices relative to extended tile)
                crop_top = h_start - h0
                crop_left = w_start - w0
                crop_bottom = crop_top + h_size
                crop_right = crop_left + w_size

                y_tile = y_tile[:, :,
                                crop_top:crop_bottom,
                                crop_left:crop_right]  # N x C_out x h_size x w_size

                w_row_out.append(y_tile)
                w_start = w_end
            tiles_out.append(torch.cat(w_row_out, dim=-1))  # concat along W
            h_start = h_end

        y = torch.cat(tiles_out, dim=-2)  # concat along H
        return y

    @staticmethod
    def _compute_splits(L: int, n_splits: int):
        """Split length L into n_splits segments that differ by at most 1."""
        base = L // n_splits
        rem = L % n_splits
        splits = []
        for i in range(n_splits):
            splits.append(base + (1 if i < rem else 0))
        return splits
