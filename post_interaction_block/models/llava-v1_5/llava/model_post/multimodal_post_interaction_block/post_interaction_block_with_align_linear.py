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
from .configuration_post_interaction_block import LlamaPIBConfig


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


class PIBSelfAttention(nn.Module):
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

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

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


class PIBCrossAttention(nn.Module):

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

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

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


class PIBLinear(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        # self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, pretraining_tp=1) -> torch.Tensor:
        
        if pretraining_tp > 1:
            fc1_slices = self.fc1.weight.split(self.vocabintermediate_size_size // self.mm_hidden_size, dim=0)
            hidden_states = [F.linear(hidden_states, fc1_slices[i]) for i in range(pretraining_tp)]
            hidden_states = torch.cat(hidden_states, dim=-1)
        else:
            hidden_states = self.fc1(hidden_states)
            
        # hidden_states = self.activation_fn(hidden_states)
        
        # if pretraining_tp > 1:
        #     fc2_slices = self.fc2.weight.split(self.vocabintermediate_size_size // self.mm_hidden_size, dim=0)
        #     hidden_states = [F.linear(hidden_states, fc2_slices[i]) for i in range(pretraining_tp)]
        #     hidden_states = torch.cat(hidden_states, dim=-1)
        # else:
        #     hidden_states = self.fc2(hidden_states)
        
        return hidden_states


class PIBCATransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, hidden_act, attention_dropout=0.0, layer_norm_eps=1e-5):
        super().__init__()
        self.embed_dim = hidden_size
        self.cross_attn = PIBCrossAttention(hidden_size, num_attention_heads, attention_dropout=attention_dropout)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
        self.mlp = PIBMLP(hidden_size, intermediate_size, hidden_act)
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

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        
        hidden_states, attn_weights = self.cross_attn(
            image_features=image_features,
            hidden_states=hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, pretraining_tp=1)
        hidden_states = residual + hidden_states

        return hidden_states


class PostInteractionBlock(nn.Module):
    
    def __init__(self, config: LlamaPIBConfig, depth=1):
        super().__init__()
        """
        copy config from configuration_llama.py LlamaConfig
        """
        self.config = config
        
        # self.align = AlignMLP(config.mm_hidden_size, config.intermediate_size, config.hidden_size, config.align_hidden_act)
        
        self.image_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.blocks = nn.ModuleList([
            PIBCATransformerBlock(config.hidden_size, config.num_attention_heads, config.intermediate_size, config.hidden_act)
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
        image_features = self.image_norm(image_features)
        
        attention_mask = None
        
        for blk in self.blocks:
            outputs = blk(image_features, hidden_states, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, causal_attention_mask, pretraining_tp=self.config.pretraining_tp)
            
        # _outputs = outputs
            
        outputs = outputs + hidden_states
        
        
        
        # # Next For loop is for calculating R-square values for each sample in the batch
        # # Calculate R-square for each sample in the batch
        # batch_size = hidden_states.size(0)
        # r_square_values = []
        
        # for i in range(batch_size):
        #     # Select current sample
        #     y_true = hidden_states[i]         # (seq_length, hidden_size)
        #     y_pred = _outputs[i]               # (seq_length, hidden_size)
            
        #     # 计算所有元素的均值作为 mean_y_true
        #     # # Calculate means along the seq_length dimension
        #     # mean_y_true = torch.mean(y_true, dim=0, keepdim=True)  # (1, hidden_size)
        #     # mean_y_true = torch.mean(mean_y_true, dim=1, keepdim=True).flatten(-1)  # (1, hidden_size)
            
        #     # # Calculate numerator and denominator
        #     # numerator = torch.sum((y_true - y_pred)**2)
        #     # denominator = torch.sum((y_true - mean_y_true)**2)
            
        #     # # Calculate R-square value
        #     # r_square = 1 - (numerator / denominator)
        #     # r_square_values.append(r_square.item())
            
        #     # 计算某一维度上的均值作为 mean_y_true
        #     y_true = torch.mean(y_true, dim=0, keepdim=True)
        #     y_pred = torch.mean(y_pred, dim=0, keepdim=True)
            
        #     mean_y_true = torch.mean(y_true, dim=1, keepdim=True)
            
        #     numerator = torch.sum((y_true - y_pred)**2)
        #     denominator = torch.sum((y_true - mean_y_true)**2)
            
        #     r_square = 1 - (numerator / denominator)
        #     r_square_values.append(r_square.item())
        #     print("R-square value:", r_square.item())
        
        # # Convert r_square_values to tensor and print
        # r_square_tensor = torch.tensor(r_square_values)
        # print("R-square values:", r_square_tensor)
        
        return outputs