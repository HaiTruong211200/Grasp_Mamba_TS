#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from transformers import AutoModel
import math


class TextImageFusionBlock(nn.Module):
    def __init__(self, image_channels, text_channels, output_channels):
        super().__init__()
        self.conv_img = nn.Conv2d(image_channels, output_channels, kernel_size=1)
        self.conv_text = nn.Conv2d(text_channels, output_channels, kernel_size=1)
        self.conv_fusion = nn.Conv2d(output_channels * 2, output_channels, kernel_size=3, padding=1)
        
    def forward(self, img_features, text_features):
        B, _, H, W = img_features.shape
        text_features = text_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        
        img_proj = self.conv_img(img_features)
        text_proj = self.conv_text(text_features)
        
        fused = torch.cat([img_proj, text_proj], dim=1)
        return self.conv_fusion(fused)

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        return self.upsample(self.conv(x))
    
    
class Head(nn.Module):
    def __init__(self, in_channels=1024, image_size=224):
        super().__init__()
        
        self.out_height = image_size
        self.out_width = image_size
        
        # Reshape and upsample
        self.reshape = nn.Conv2d(in_channels, 512, kernel_size=1)  # Adjust for (8, 2048, 13, 13)
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Final convolutions to get to the desired output size
        self.final_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Output layers
        self.pos_output = nn.Conv2d(32, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(32, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(32, 1, kernel_size=1)
        self.width_output = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (batch_size, 2048, 13, 13)
        x = self.reshape(x)  # Shape: (batch_size, 512, 13, 13)
        x = self.upsample(x)  # Shape: (batch_size, 32, 208, 208)
        x = F.interpolate(x, size=(self.out_height, self.out_width), mode='bilinear', align_corners=False)  # Shape: (batch_size, 32, 416, 416)
        x = self.final_conv(x)  # Shape: (batch_size, 32, 416, 416)
        
        pos_output = self.pos_output(x)  # Shape: (batch_size, 1, 416, 416)
        cos_output = self.cos_output(x)  # Shape: (batch_size, 1, 416, 416)
        sin_output = self.sin_output(x)  # Shape: (batch_size, 1, 416, 416)
        width_output = self.width_output(x)  # Shape: (batch_size, 1, 416, 416)
        
        return pos_output, cos_output, sin_output, width_output
    
class GraspMamba(nn.Module):
    """
    GraspMamba,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 device=None,
                 resolution=416,
                 channel_size=32,
                 output_channels=1,
                 dropout=False,
                 prob=0.1,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.num_query = 25
        # Calculate new spatial dimensions
        self.new_spatial_dim = int(math.sqrt(self.num_query))
        assert self.new_spatial_dim ** 2 == self.num_query, "num_query must be a perfect square"

        # Text encoder
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.model = AutoModel.from_pretrained("nvidia/MambaVision-S-1K", trust_remote_code=True)
        # eval mode for inference
        self.model.cuda().eval()
      
        # Head and MLP
        self.output = Head(in_channels=768, image_size=resolution)
        

    def forward(self, img, text):
        # Mamba vision
        _, image_features = self.model(img)
        
        x = image_features[-1]
        # Grasp output
        pos_output, cos_output, sin_output, width_output = self.output(x)

        return pos_output, cos_output, sin_output, width_output

  
        




