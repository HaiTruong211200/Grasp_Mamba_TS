#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import math
from inference.models.segmentation import SegHead


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

    def __init__(self, resolution=416, segmentation=False,
                 vision_model="nvidia/MambaVision-L-1K",
                 text_model="google/siglip-base-patch16-224"):
        """
        Args:
            resolution: input resolution.
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.num_query = 25
        # Calculate new spatial dimensions
        self.new_spatial_dim = int(math.sqrt(self.num_query))
        assert self.new_spatial_dim ** 2 == self.num_query, "num_query must be a perfect square"

         # Replace CLIP text encoder with SigLIP text encoder
        self.siglip_model = AutoModel.from_pretrained(text_model)
        self.siglip_processor = AutoProcessor.from_pretrained(text_model)
        self.siglip_model.eval()
        for param in self.siglip_model.parameters():
            param.requires_grad = False
        
        self.model = AutoModel.from_pretrained(vision_model, trust_remote_code=True)
        # Freeze or fine-tuning
        self.model.train()
        # self.model.cuda().eval()
        # for param in self.model.parameters():
        #     param.requires_grad = True

        # Text-Image Fusion blocks
        self.fusion_stage4 = TextImageFusionBlock(1568, 768, 512)
        self.fusion_stage3 = TextImageFusionBlock(784, 768, 384)
        self.fusion_stage2 = TextImageFusionBlock(392, 768, 192)
        self.fusion_stage1 = TextImageFusionBlock(196, 768, 96)
        
        # Upscale blocks
        self.upscale3 = UpscaleBlock(512, 384)
        self.upscale2 = UpscaleBlock(384, 192)
        self.upscale1 = UpscaleBlock(192, 96)

        # Combination layers
        self.conv = nn.Conv2d(96, 1024, kernel_size=1)

        # Head and MLP
        self.output = Head(in_channels=1024, image_size=resolution)
        self.seg_head = SegHead(96) if segmentation else None
        

    def train(self, mode=True):
        super().train(mode)
        self.siglip_model.eval()
        return self

    def forward(self, img, text, return_seg=False):
        if return_seg and self.seg_head is None:
            raise ValueError('Segmentation head is disabled; construct with segmentation=True')
        # Encode text using CLIP
        with torch.no_grad():
            text_inputs = self.siglip_processor(
                text=text,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            text_inputs = {k: v.to(img.device) for k, v in text_inputs.items()}
            text_features = self.siglip_model.get_text_features(**text_inputs).float()
        # print("text_features",text_features.shape)
        # Mamba vision + text
        _, image_features = self.model(img)
        
        stage1, stage2, stage3, stage4 = image_features
        # print("stage1",stage1.shape)
        # print("stage2",stage2.shape)
        # print("stage3",stage3.shape)
        # print("stage4",stage4.shape)
        # Multi-scale fusion and upsampling
        fused4 = self.fusion_stage4(stage4, text_features)
        
        up3 = F.interpolate(self.upscale3.conv(fused4), size=stage3.shape[-2:], mode='bilinear', align_corners=False)
        fused3 = self.fusion_stage3(stage3, text_features)
        combined3 = up3 + fused3
        
        up2 = F.interpolate(self.upscale2.conv(combined3), size=stage2.shape[-2:], mode='bilinear', align_corners=False)
        fused2 = self.fusion_stage2(stage2, text_features)
        combined2 = up2 + fused2
        
        up1 = F.interpolate(self.upscale1.conv(combined2), size=stage1.shape[-2:], mode='bilinear', align_corners=False)
        fused1 = self.fusion_stage1(stage1, text_features)
        combined1 = up1 + fused1
        
        # Final processing
        final_features = F.relu(self.conv(combined1))

        # Grasp output
        pos_output, cos_output, sin_output, width_output = self.output(final_features)

        outputs = (pos_output, cos_output, sin_output, width_output)
        if return_seg:
            return outputs + (self.seg_head(combined1, img.shape[-2:]),)
        return outputs

  
        




