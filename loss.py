import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, reg_type, device, num_patches):
        super(Loss, self).__init__()
        self.reg_type = reg_type
        self.device = device
        self.num_patches = num_patches
        self.reg_weights = {"KL_divergence":0.01, "L2_norm":0.1, "soft_R":0.1, "R":0.1}
        self.reg_hypers = {"R":{"tau":0.01}}

    def forward(self, scores, masks, attn_weights=None):
        loss_per_patch = F.binary_cross_entropy_with_logits(scores, masks, reduction='none')
        loss_per_image = loss_per_patch.sum(dim=1)

        # zeros_count = (masks == 0).sum(dim=1)
        # non_zeros_count = (masks != 0).sum(dim=1)

        # anomalous_loss = (masks * loss_per_patch).sum(dim=1) / (non_zeros_count + 1e-8)
        # normal_loss = ((1 - masks) * loss_per_patch).sum(dim=1) / (zeros_count + 1e-8)

        # loss_per_image = normal_loss.sum() + anomalous_loss.sum()

        reg_term = 0
        if attn_weights is not None:
            reg_term = self.reg(attn_weights)  

        total_loss = loss_per_image.mean() + 1 * reg_term
        return total_loss

    def reg(self, attn_weights):
        reg_term = 0
        coefs = [0.005, 0.01, 0.05]
        if self.reg_type == "KL_divergence":
            for i, attn_weight in enumerate(attn_weights):
                reg_term += coefs[i] * F.kl_div(torch.log(attn_weight + 1e-8),  torch.full_like(attn_weight, 1 / self.num_patches).to(self.device), reduction='mean')
        
        # not recommended        
        elif self.reg_type == "L2_norm":
            for attn_weight in attn_weights:
                reg_term += attn_weight.norm(p='fro')
        elif self.reg_type == "soft_R":
            for attn_weight in attn_weights:
                soft_mask = torch.sigmoid((self.delta - attn_weight) / self.reg_hypers["R"]["tau"])  # Smooth threshold
                reg_term += 1 / (torch.sum(attn_weight * soft_mask) + 1e-8)
                #reg_term -= torch.sum(attn_weight * soft_mask)
        elif self.reg_type == "R":
            for attn_weight in attn_weights:
                mask = (attn_weight <= 1 / self.num_patches).float() # Hard threshold
                reg_term += 1 / (torch.sum(attn_weight * mask) + 1e-8) # Summing selected attention values
                #reg_term -= torch.sum(attn_weight * mask) # Summing selected attention values

        
        return reg_term