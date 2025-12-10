"""
Custom trainer implementations.
"""

from typing import Any, Optional, final, Union, override

import torch
from torch import nn
from transformers.trainer import Trainer

from .losses import FocalLoss, FocalLossAlpha, FocalLossGamma, TaskType


@final
class FocalLossTrainer(Trainer):
    """
    Custom Trainer that uses Focal Loss instead of standard CrossEntropyLoss.

    Useful for handling class imbalance in token classification tasks.
    """

    def __init__(
        self,
        *args,
        focal_loss_alpha: FocalLossAlpha = 1.0,
        focal_loss_gamma: FocalLossGamma = 2.0,
        task_type: TaskType,
        num_labels: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Initialize focal loss with provided parameters
        self.focal_loss = FocalLoss(
            gamma=focal_loss_gamma,
            alpha=focal_loss_alpha,
            task_type=task_type,
            num_labels=num_labels,
        )
        self.num_labels = num_labels

        print(
            f"Initialized FocalLoss with alpha={focal_loss_alpha}, gamma={focal_loss_gamma}"
        )

    @override
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        Override compute_loss to use Focal Loss instead of default CrossEntropyLoss.
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is None:
            raise ValueError("Labels are required for computing loss")

        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        loss = self.focal_loss(active_logits, active_labels)

        return (loss, outputs) if return_outputs else loss
