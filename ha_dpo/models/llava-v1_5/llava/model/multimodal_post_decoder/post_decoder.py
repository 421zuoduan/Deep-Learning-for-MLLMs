#!/usr/bin/env python 
# -*- coding:utf-8 -*- 
'''
 * @Author: Ruochen Cui 
 * @Date: 2024-03-27 17:34:51 
 * @Last Modified by:   Ruochen Cui 
 * @Last Modified time: 2024-03-27 17:34:51 
 * @Desc: 
'''
from typing import Any, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from .configuration_post_decoder import LlavaLlamaPostDecoderConfig


# Copied from transformers.models.bart.modeling_bart._expand_mask
def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

# Copied from transformers.models.blip_2.modeling_clip.get_extended_attention_mask
def get_extended_attention_mask(
    self,
    attention_mask: torch.Tensor,
    input_shape: Tuple[int],
    device: torch.device,
    has_query: bool = False,
) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
        device (`torch.device`):
            The device of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape
            )
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Revised from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    # revised to no causal mask
    # if input_shape[-1] > 1:
    #     combined_attention_mask = make_causal_mask(
    #         input_shape,
    #         inputs_embeds.dtype,
    #         device=inputs_embeds.device,
    #         past_key_values_length=past_key_values_length,
    #     )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        # combined_attention_mask = (
        #     expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        # )

    return expanded_attn_mask


class AlignMLP(nn.Module):
    def __init__(self, mm_hidden_size, intermediate_size, hidden_size, hidden_act):
        super().__init__()

        self.mm_hidden_size = mm_hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(mm_hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, pretraining_tp=1) -> torch.Tensor:
        
        # hidden_states = self.fc1(hidden_states)
        # hidden_states = self.activation_fn(hidden_states)
        # hidden_states = self.fc2(hidden_states)
        # return hidden_states
        
        if pretraining_tp > 1:
            fc1_slices = self.fc1.weight.split(self.vocabintermediate_size_size // self.mm_hidden_size, dim=0)
            hidden_states = [F.linear(hidden_states, fc1_slices[i]) for i in range(pretraining_tp)]
            hidden_states = torch.cat(hidden_states, dim=-1)
        else:
            hidden_states = self.fc1(hidden_states)
            
        hidden_states = self.activation_fn(hidden_states)
        
        if pretraining_tp > 1:
            fc2_slices = self.fc2.weight.split(self.vocabintermediate_size_size // self.mm_hidden_size, dim=0)
            hidden_states = [F.linear(hidden_states, fc2_slices[i]) for i in range(pretraining_tp)]
            hidden_states = torch.cat(hidden_states, dim=-1)
        else:
            hidden_states = self.fc2(hidden_states)
        
        return hidden_states


class PostDecoderSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout=0.0):
        super().__init__()

        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
            
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class PostDecoderCrossAttention(nn.Module):

    # attention_dropout=0.0 copied from configuration_clip.py
    def __init__(self, hidden_size, num_attention_heads, attention_dropout=0.0):
        super().__init__()

        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        image_features: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(image_features), -1, bsz)
        value_states = self._shape(self.v_proj(image_features), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class PostDecoderMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, pretraining_tp=1) -> torch.Tensor:
        
        
        # hidden_states = self.fc1(hidden_states)
        # hidden_states = self.activation_fn(hidden_states)
        # hidden_states = self.fc2(hidden_states)
        # return hidden_states
        
        if pretraining_tp > 1:
            fc1_slices = self.fc1.weight.split(self.vocabintermediate_size_size // self.mm_hidden_size, dim=0)
            hidden_states = [F.linear(hidden_states, fc1_slices[i]) for i in range(pretraining_tp)]
            hidden_states = torch.cat(hidden_states, dim=-1)
        else:
            hidden_states = self.fc1(hidden_states)
            
        hidden_states = self.activation_fn(hidden_states)
        
        if pretraining_tp > 1:
            fc2_slices = self.fc2.weight.split(self.vocabintermediate_size_size // self.mm_hidden_size, dim=0)
            hidden_states = [F.linear(hidden_states, fc2_slices[i]) for i in range(pretraining_tp)]
            hidden_states = torch.cat(hidden_states, dim=-1)
        else:
            hidden_states = self.fc2(hidden_states)
        
        return hidden_states


class PostDecoderSATransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, hidden_act, attention_dropout=0.0, layer_norm_eps=1e-5):
        super().__init__()
        
        self.embed_dim = hidden_size
        self.self_attn = PostDecoderSelfAttention(hidden_size, num_attention_heads, attention_dropout=attention_dropout)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
        self.mlp = PostDecoderMLP(hidden_size, intermediate_size, hidden_act=hidden_act)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)

    def forward(
        self,
        image_features: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        
        batch_size = attention_mask.size(0)
        seq_length = attention_mask.size(1)
        past_key_values_length = hidden_states.size(1) - image_features.size(1)
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class PostDecoderCATransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, hidden_act, attention_dropout=0.0, layer_norm_eps=1e-5):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = PostDecoderSelfAttention(hidden_size, num_attention_heads, attention_dropout=attention_dropout)
        self.cross_attn = PostDecoderCrossAttention(hidden_size, num_attention_heads, attention_dropout=attention_dropout)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
        self.mlp = PostDecoderMLP(hidden_size, intermediate_size, hidden_act)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)

    def forward(
        self,
        image_features: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]],
        inputs_embeds: Optional[torch.FloatTensor],
        causal_attention_mask: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        pretraining_tp=1,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states, attn_weights = self.cross_attn(
            image_features=image_features,
            hidden_states=hidden_states,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, pretraining_tp=1)
        hidden_states = residual + hidden_states

        return hidden_states


class PostDecoder(nn.Module):
    
    def __init__(self, config: LlavaLlamaPostDecoderConfig, depth=1):
        super().__init__()
        """
        copy config from configuration_llama.py LlamaConfig
        
        vocab_size=32000,
        hidden_size=4096,
        mm_hidden_size=1024,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        align_hidden_act="gelu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        """
        self.config = config
        
        self.align = AlignMLP(config.mm_hidden_size, config.intermediate_size, config.hidden_size, config.align_hidden_act)
        
        self.blocks = nn.ModuleList([
            PostDecoderCATransformerBlock(config.hidden_size, config.num_attention_heads, config.intermediate_size, config.hidden_act)
            for i in range(depth)])
        
    def forward(self, image_features, hidden_states, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, causal_attention_mask=None, pretraining_tp=1):
        """_summary_

        Args:
            image_features (Tensor): torch.Size([8, 576, 1024])
            hidden_states (Tensor): torch.Size([8, 768, 4096]) 770 767 768

        Returns:
            outputs (Tensor): torch.Size([8, 752, 4096])
        """
    
    
        image_features = self.align(image_features, pretraining_tp=self.config.pretraining_tp)
        
        # copied from LlmaModel.forward
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        attention_mask = torch.ones((batch_size, 1, seq_length, seq_length), device=device, dtype=torch.bool)
        
        
        for blk in self.blocks:
            outputs = blk(image_features, hidden_states, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, causal_attention_mask, pretraining_tp=self.config.pretraining_tp)
            
        outputs = outputs + hidden_states
        
        return outputs