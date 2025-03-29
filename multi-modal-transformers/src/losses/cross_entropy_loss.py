import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss for classification and generation tasks.
    Supports label smoothing and class weighting.
    """
    
    def __init__(
        self,
        num_classes: Optional[int] = None,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize cross entropy loss.
        
        Args:
            num_classes: Number of classes for classification
            weight: Class weights for imbalanced datasets
            ignore_index: Index to ignore in the target (e.g., padding)
            reduction: Reduction method (mean, sum, or none)
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute cross entropy loss.
        
        Args:
            outputs: Model outputs containing logits
            batch: Batch data containing targets
            
        Returns:
            Loss value
        """
        # Get logits from outputs
        if "logits" in outputs:
            logits = outputs["logits"]
        else:
            raise ValueError("Outputs must contain 'logits'")
        
        # Get targets from batch
        if "labels" in batch:
            targets = batch["labels"]
        elif "answers" in batch:
            targets = batch["answers"]
        elif "caption_ids" in batch:
            targets = batch["caption_ids"]
        else:
            raise ValueError("Batch must contain 'labels', 'answers', or 'caption_ids'")
        
        # Handle different task types
        if logits.dim() == 2:
            # Classification task: [batch_size, num_classes]
            return self.classification_loss(logits, targets)
        elif logits.dim() == 3:
            # Generation task: [batch_size, seq_len, vocab_size]
            return self.generation_loss(logits, targets)
        else:
            raise ValueError(f"Unsupported logits dimension: {logits.dim()}")
    
    def classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for classification tasks.
        
        Args:
            logits: Predicted logits [batch_size, num_classes]
            targets: Target class indices [batch_size]
            
        Returns:
            Loss value
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )
    
    def generation_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for generation tasks.
        
        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            targets: Target token indices [batch_size, seq_len]
            
        Returns:
            Loss value
        """
        # Reshape logits for cross-entropy calculation
        batch_size, seq_len, vocab_size = logits.size()
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )


class FocalLoss(nn.Module):
    """
    Focal loss for handling imbalanced classification problems.
    Applies a modulating factor to the standard cross-entropy loss.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Class weights for imbalanced datasets
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted
            reduction: Reduction method (mean, sum, or none)
            ignore_index: Index to ignore in the target (e.g., padding)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            outputs: Model outputs containing logits
            batch: Batch data containing targets
            
        Returns:
            Loss value
        """
        # Get logits from outputs
        if "logits" in outputs:
            logits = outputs["logits"]
        else:
            raise ValueError("Outputs must contain 'logits'")
        
        # Get targets from batch
        if "labels" in batch:
            targets = batch["labels"]
        elif "answers" in batch:
            targets = batch["answers"]
        else:
            raise ValueError("Batch must contain 'labels' or 'answers'")
        
        # Compute focal loss
        return self.focal_loss(logits, targets)
    
    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Predicted logits [batch_size, num_classes]
            targets: Target class indices [batch_size]
            
        Returns:
            Loss value
        """
        # Compute softmax probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Create one-hot encoding of targets
        num_classes = logits.size(-1)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(-1, targets.unsqueeze(-1), 1)
        
        # Apply mask for ignored indices
        mask = (targets != self.ignore_index).float().unsqueeze(-1)
        one_hot = one_hot * mask
        
        # Compute focal loss
        pt = (one_hot * probs).sum(-1)  # Get the probability of the target class
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t
        
        # Compute loss
        loss = -focal_weight * log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        # Apply mask for ignored indices
        loss = loss * mask.squeeze(-1)
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.sum() / mask.sum()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss 