import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for image-text retrieval tasks.
    Supports both InfoNCE and triplet margin loss formulations.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.2,
        max_violation: bool = True,
        reduction: str = "mean",
        loss_type: str = "infonce",  # Options: infonce, triplet
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for InfoNCE loss
            margin: Margin for triplet loss
            max_violation: Whether to use hard negative mining
            reduction: Reduction method (mean or sum)
            loss_type: Type of contrastive loss (infonce or triplet)
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.max_violation = max_violation
        self.reduction = reduction
        self.loss_type = loss_type
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            outputs: Model outputs containing similarity scores
            batch: Batch data
            
        Returns:
            Loss value
        """
        # Get similarity scores from outputs
        if "scores" in outputs:
            scores = outputs["scores"]
        elif "text_embeds" in outputs and "image_embeds" in outputs:
            # Compute similarity scores if not provided
            text_embeds = outputs["text_embeds"]
            image_embeds = outputs["image_embeds"]
            
            # Normalize embeddings
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
            
            # Compute similarity scores
            scores = torch.matmul(text_embeds, image_embeds.transpose(0, 1))
        else:
            raise ValueError("Outputs must contain either 'scores' or both 'text_embeds' and 'image_embeds'")
        
        # Compute loss based on the specified type
        if self.loss_type == "infonce":
            return self.infonce_loss(scores)
        elif self.loss_type == "triplet":
            return self.triplet_loss(scores)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def infonce_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            scores: Similarity scores [batch_size, batch_size]
            
        Returns:
            Loss value
        """
        batch_size = scores.size(0)
        
        # Labels are the diagonal indices (matching pairs)
        labels = torch.arange(batch_size, device=scores.device)
        
        # Scale scores by temperature
        scores = scores / self.temperature
        
        # Compute image-to-text and text-to-image losses
        i2t_loss = F.cross_entropy(scores, labels)
        t2i_loss = F.cross_entropy(scores.t(), labels)
        
        # Combine losses
        loss = (i2t_loss + t2i_loss) / 2
        
        return loss
    
    def triplet_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet margin loss.
        
        Args:
            scores: Similarity scores [batch_size, batch_size]
            
        Returns:
            Loss value
        """
        batch_size = scores.size(0)
        diagonal = scores.diag().view(batch_size, 1)
        
        # Compute cost matrices for image-to-text and text-to-image
        # For each row (image), the diagonal element is the positive pair
        # and all other elements in the row are negative pairs
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        
        # Compute margins: positive - negative + margin
        # We want positive scores to be higher than negative scores by at least the margin
        cost_i2t = (self.margin + scores - d1).clamp(min=0)
        cost_t2i = (self.margin + scores - d2).clamp(min=0)
        
        # Clear diagonals (positive pairs)
        mask = torch.eye(batch_size, device=scores.device) > 0.5
        cost_i2t = cost_i2t.masked_fill(mask, 0)
        cost_t2i = cost_t2i.masked_fill(mask, 0)
        
        # Hard negative mining if enabled
        if self.max_violation:
            cost_i2t = cost_i2t.max(1)[0]
            cost_t2i = cost_t2i.max(0)[0]
        
        # Apply reduction
        if self.reduction == "sum":
            loss = cost_i2t.sum() + cost_t2i.sum()
        else:  # mean
            loss = cost_i2t.mean() + cost_t2i.mean()
        
        return loss 