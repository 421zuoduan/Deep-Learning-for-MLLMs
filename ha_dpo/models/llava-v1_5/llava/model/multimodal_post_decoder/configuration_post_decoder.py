/*
 * @Author: Ruochen Cui 
 * @Date: 2024-03-25 01:39:37 
 * @Last Modified by: Ruochen Cui
 * @Last Modified time: 2024-03-25 02:04:38
 */
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import copy
import os

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig

logger = logging.get_logger(__name__)

CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json",
    # See all CLIP models at https://huggingface.co/models?filter=clip
}


class PostDecoderBackboneConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PostDecoderModel`]. It is used to instantiate a Post Decoder
    text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the text encoder of the Post Decoder architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:

    Example:

    """
    model_type = "postdecoder_backbone_model"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the backbone config dict if we are loading from PostDecoderConfig
        if config_dict.get("model_type") == "postdecoder":
            config_dict = config_dict["postdecoder_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class PostDecoderVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PostDecoderModel`]. It is used to instantiate a Post Decoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the post decoder architecture.
    
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from [`PretrainedConfig`] for more information.
    
    Args: 
    
    Example:

    """
    
    model_type = "post_decoder_vision_model"
    
    def __init__(self, **kwargs, ):
        
        super().__init__(**kwargs)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        r"""
        Instantiate a `PostDecoderVisionConfig` (or a derived class) from a pre-trained model configuration.
        """
        
        cls._set_token_in_kwargs(kwargs)
        
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        
        # get the vision config dict if we are loading from PostDecoderConfig
        if config_dict.get("model_type") == "postdecoder":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
        
        
class PostDecoderConfig(PretrainedConfig):
    r"""
    [`PostDecoderConfig`] is the configuration class to store the configuration of a [`PostDecoderModel`]. It is used to instantiate
    a Post Decoder model according to the specified arguments, defining the post processing model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the post decoder architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`PostDecoderBackboneConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`PostDecoderVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original Post Decoder implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    """

    model_type = "PostDecoder"
    is_composition = True

    def __init__(
        self, backbone_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        backbone_config_dict = kwargs.pop("backbone_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[backbone|vision]_config_dict` to `[backbone|vision]_config`, we use the values in
        # `[backbone|vision]_config_dict` to update the values in `[backbone|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2` which is from clip.
        if backbone_config_dict is not None:
            if backbone_config is None:
                backbone_config = {}

            # This is the complete result when using `backbone_config_dict`.
            _backbone_config_dict = PostDecoderBackboneConfig(**backbone_config_dict).to_dict()

            # Give a warning if the values exist in both `_backbone_config_dict` and `backbone_config` but being different.
            for key, value in _backbone_config_dict.items():
                if key in backbone_config and value != backbone_config[key] and key not in ["transformers_version"]:
                    # If specified in `backbone_config_dict`
                    if key in backbone_config_dict:
                        message = (
                            f"`{key}` is found in both `backbone_config_dict` and `backbone_config` but with different values. "
                            f'The value `backbone_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`backbone_config_dict` is provided which will be used to initialize `PostDecoderBackboneConfig`. The "
                            f'value `backbone_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `backbone_config` with the ones in `_backbone_config_dict`.
            backbone_config.update(_backbone_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = PostDecoderVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key not in ["transformers_version"]:
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `PostDecoderVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overriden.'
                        )
                    logger.warning(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if backbone_config is None:
            backbone_config = {}
            logger.info("`backbone_config` is `None`. Initializing the `PostDecoderBackboneConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `PostDecoderVisionConfig` with default values.")

        self.backbone_config = PostDecoderBackboneConfig(**backbone_config)
        self.vision_config = PostDecoderVisionConfig(**vision_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_backbone_vision_configs(cls, backbone_config: PostDecoderBackboneConfig, vision_config: PostDecoderVisionConfig, **kwargs):
        r"""
        Instantiate a [`PostDecoderConfig`] (or a derived class) from the backbone model configuration and the vision model
        configuration.

        Returns:
            [`PostDecoderConfig`]: An instance of a configuration object
        """

        return cls(backbone_config=backbone_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["backbone_config"] = self.backbone_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output        
        
        

        


class PostDecoderModel(nn.Module):
    config_class = PostDecoderConfig
    main_input_name = "backbone_output_and_image_features"
    
    def __init__(self, config: PostDecoderConfig):
        super().__init__(config)
        self.post_decoder = PostDecoderTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_input(self) -> nn.Module:
        return self.post_decoder

    from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values=None,
        pixel_mask=None,
        return_dict=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        pass




class PostDecoderModule(nn.Module):
    def __init__(self, post_decoder, args, delay_load=False):
        
        self.is_loaded = False
        
        self.post_decoder_name = post_decoder
        
        if not delay_load
            self.load_model()
        else:
            self.cfg_only = PostDecoderConfig.from_pretrained(self.post_decoder_name)
        
        
    def load_model(self):
        self.
        self.post_decoder = LlamaModel.from_pretrained(self.post_decoder_name)
        self.post_decoder.requires_grad_(False)
        
        self.is_loaded = True


    @torch.no_grad()
    def forward(self, images):
        pass

