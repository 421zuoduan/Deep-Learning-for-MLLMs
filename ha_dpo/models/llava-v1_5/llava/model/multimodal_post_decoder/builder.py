/*
 * @Author: Ruochen Cui 
 * @Date: 2024-03-25 15:37:12 
 * @Last Modified by:   Ruochen Cui 
 * @Last Modified time: 2024-03-25 15:37:12 
 */
import os
from .post_decoder import PostDecoder


def build_post_decoder(post_decoder_cfg, **kwargs):
    post_decoder = getattr(post_decoder_cfg, 'mm_post_decoder', getattr(post_decoder_cfg, 'post_decoder', None))
    is_absolute_path_exists = os.path.exists(post_decoder)
    if is_absolute_path_exists or post_decoder.startswith("openai") or post_decoder.startswith("laion"):
        return PostDecoder(post_decoder, args=post_decoder_cfg, **kwargs)

    raise ValueError(f'Unknown post decoder: {post_decoder}')