from .crossentropy import CrossEntropyLoss
from .dice import DiceLoss, dice_loss
from .umap import UmapLoss, umap_loss
from .boundary_aware import BoundaryAwareLoss   # ← 新增

__all__ = [
    "CrossEntropyLoss",
    "DiceLoss",
    "dice_loss",
    "UmapLoss",
    "umap_loss",
    "BoundaryAwareLoss",
]
