/*
 * @Author: Ruochen Cui 
 * @Date: 2024-03-25 12:27:38 
 * @Last Modified by: Ruochen Cui
 * @Last Modified time: 2024-03-25 12:28:04
 */
import torch
import torch.nn as nn

from .configuration_post_decoder import PostDecoderConfig
from .modeling_post_decoder import PostDecoderVision


# Post Decoder Model
class PostDecoder(nn.Module):
    def __init__(self, post_decoder, args, delay_load=False):
        super().__init__()
        
        self.is_loaded = False
        
        self.post_decoder_name = post_decoder
        self.select_layer = args.mm_vision_select_layer

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = PostDecoderConfig.from_pretrained(self.post_decoder_name)
            
    def get_vision_tower_output(self, ):
        pass
        
    def get_backbone_output(self, ):
        pass

    def load_model(self, ):
        r"""
        加载 model
        """
        
        

    
    def feature_select(self, image_forward_outs):
        pass
        
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # TODO: 补充post decoder参数
                image_forward_out = self.post_decoder()
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.apped(image_feature)
        else:
            # TODO: 补充post decoder参数
            image_forward_outs = self.post_decoder()
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
                
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2