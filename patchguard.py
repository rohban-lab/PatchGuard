import re
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x

    return my_forward

class FeatureExtractor(nn.Module):
    def __init__(self, hf_path, feature_layer_indices, reg_layer_indices, image_size, device):
        super(FeatureExtractor, self).__init__()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mu = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)
        self.norm = lambda x: (x - self.mu) / self.std

        self.feature_layer_indices = feature_layer_indices
        self.reg_layer_indices = reg_layer_indices

        self.pretrained_model = timm.create_model(hf_path, pretrained=False, num_classes=0, img_size=image_size).to(device)

        self.embed_dim = len(feature_layer_indices) * self.pretrained_model.embed_dim
        self.patch_size = self.pretrained_model.patch_embed.patch_size[0]
        self.num_patches = (image_size // self.patch_size) ** 2

        pattern = r'reg(\d+)'
        match = re.search(pattern, hf_path)
        self.start_index = int(match.group(1)) + 1 if match else 1

        indices = set(feature_layer_indices + reg_layer_indices)
        for i in indices:
            self.pretrained_model.blocks[i-1].attn.forward = my_forward_wrapper(self.pretrained_model.blocks[i-1].attn)
            
    def forward(self, x, use_reg=True):
        x = self.norm(x)
        x = self.pretrained_model.patch_embed(x)
        x = self.pretrained_model._pos_embed(x)
        x = self.pretrained_model.patch_drop(x)
        x = self.pretrained_model.norm_pre(x)

        out = []
        attention_weights = []

        # iterating through the layers up to last layer to extract from => 12 layers
        for idx, layer in enumerate(self.pretrained_model.blocks, start=1):
            x = layer(x)

            if idx in self.feature_layer_indices:
                features_layer = self.pretrained_model.norm(x[:, self.start_index:, :])
                out.append(features_layer)

            if use_reg and idx in self.reg_layer_indices:
                attention_map = layer.attn.attn_map
                attention_weights.append(attention_map[:, :, 1:, 1:])  # Remove CLS token

            if idx == max(self.feature_layer_indices):  
                break

        features = torch.cat(out, dim=-1)

        return (features, attention_weights) if use_reg else (features, None)
                
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.layer_norm_1(x + self.dropout1(attn_output))
        x = x + self.dropout2(self.linear(x))

        return x

class Discriminator(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_patches, num_layers=1, num_heads=12, dropout_rate=0):
        super(Discriminator, self).__init__()
        self.transformer_encoder = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout_rate) for _ in range(num_layers)])
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, 1)
        )
        self.positional_encodings = nn.Parameter(torch.randn(num_patches, embed_dim))

    def forward(self, x):
        x = x + self.positional_encodings.unsqueeze(0)
        x = self.transformer_encoder(x)
        x = self.output_layer(x).squeeze(-1)

        return x

class PatchGuard(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.feature_extractor = FeatureExtractor(args.hf_path, args.feature_layers, args.reg_layers, args.image_size, device)

        embed_dim = self.feature_extractor.embed_dim
        self.num_patches = self.feature_extractor.num_patches
        self.patch_size = self.feature_extractor.patch_size
        self.patches_per_side = int(np.sqrt(self.num_patches))

        self.discriminator = Discriminator(embed_dim, args.hidden_dim, self.num_patches, args.dsc_layers, args.dsc_heads, 0.2).to(device)

    def forward(self, x):
        embeddings, _ = self.feature_extractor(x, False)
        scores = self.discriminator(embeddings)
        return scores