import copy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from inference.models.mamba.mamba_L_Siglip import GraspMamba
import clip
from ruamel.yaml import YAML 
import inference.models.lgdm.albef.utils as utils
from inference.post_process import post_process_output
import seaborn as sns
import os
import matplotlib.pyplot as plt

class Grasp_Mamba(nn.Module):
    def __init__(self, input_size=224, vision_model='nvidia/MambaVision-L-1K',
                 text_model='google/siglip-base-patch16-224'):
        """
        Initialize the model 
        backbone: Text backbone, Image backbone and the vision mamba 
        
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.grasp_mamba = GraspMamba(resolution=input_size,
                                      vision_model=vision_model, text_model=text_model)

    def forward(self, img, query):
        #Encode text
        # For CLIP
        # text_input = clip.tokenize(query).to(self.device)
        pos_output, cos_output, sin_output, width_output = self.grasp_mamba(img, query)
        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, img, yc, query):
        y_pos, y_cos, y_sin, y_width = yc[0], yc[1], yc[2], yc[3]
        pos_pred, cos_pred, sin_pred, width_pred = self(img, query)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)
       
        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss,
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }
    
    
    def _encode_text(self, raw_text, device=None):
        # raw_text - list (batch_size length) of strings with input text prompts
        max_text_len = 20 # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return texts
        
    