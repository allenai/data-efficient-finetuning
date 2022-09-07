'''
Mix and Match Adapter using recommended settings from https://arxiv.org/abs/2110.04366
'''
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import (
    T5LayerCrossAttention,
    T5LayerSelfAttention,
    T5Block
)
from transformers import PreTrainedModel
from allennlp.models import Model

from attribution.model import BasicSeq2Seq

logger = logging.getLogger(__name__)


class Adapter(nn.Module):
    def __init__(self, adapter_size, hidden_size):
        super().__init__()
        self.adapter_input_size = hidden_size
        self.adapter_latent_size = adapter_size
        self.non_linearity = nn.ReLU()

        # down projection
        self.down_proj = nn.Linear(self.adapter_input_size, self.adapter_latent_size, bias=False)
        # up projection
        self.up_proj = nn.Linear(self.adapter_latent_size, self.adapter_input_size, bias=False)
        # layer norm
        self.ln = nn.LayerNorm(self.adapter_input_size)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights -> so that initially we the whole Adapter layer is a near-identity function"""
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.ln(self.up_proj(self.non_linearity(self.down_proj(x))))


# adapter modifies the feedforward layer
class T5LayerFFWithAdapter(nn.Module):
    def __init__(self, T5LayerFF, adapter_size, hidden_size):
        super().__init__()
        self.DenseReluDense = T5LayerFF.DenseReluDense
        self.adapter = Adapter(adapter_size, hidden_size)
        self.layer_norm = T5LayerFF.layer_norm
        self.dropout = T5LayerFF.dropout

    def forward(self, hidden_states):
        ln_hidden_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(ln_hidden_states)
        hidden_states = hidden_states + self.dropout(forwarded_states) + self.adapter(ln_hidden_states)
        return hidden_states


# prefixes modify the attention layers.
class T5AttentionPrefixTuning(nn.Module):
    def __init__(self, attention_layer, num_prefix_tokens, parameterization, shared=None):
        super().__init__()
        self.is_decoder = attention_layer.is_decoder
        self.has_relative_attention_bias = attention_layer.has_relative_attention_bias

        self.relative_attention_num_buckets = attention_layer.relative_attention_num_buckets
        self.d_model = attention_layer.d_model
        self.key_value_proj_dim = attention_layer.key_value_proj_dim
        self.n_heads = attention_layer.n_heads
        self.dropout = attention_layer.dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.prune_heads = attention_layer.prune_heads
        self._relative_position_bucket = attention_layer._relative_position_bucket
        self.compute_bias = attention_layer.compute_bias

        self.q = attention_layer.q
        self.k = attention_layer.k
        self.v = attention_layer.v
        self.o = attention_layer.o
        if self.has_relative_attention_bias:
            self.relative_attention_bias = attention_layer.relative_attention_bias
        self.pruned_heads = attention_layer.pruned_heads
        self.gradient_checkpointing = attention_layer.gradient_checkpointing

        self.parameterization = parameterization
        self.num_prefix_tokens = num_prefix_tokens
        self.mode = "apply"

        self.setup_prefix(shared)

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Modified from T5Attention forward
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        pask_key_value, query_length, use_cache disabled
        """
        assert past_key_value is None
        assert query_length is None
        assert not use_cache
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]
        key_length = seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, prefix_states):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                output_states = proj_layer(hidden_states)
            else:
                # cross-attn
                output_states = proj_layer(key_value_states)
            if prefix_states is not None:
                output_states = torch.cat([prefix_states, output_states], dim=1)
            return output_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        if self.mode == "apply":
            prefix = self.get_prefix(batch_size)
        else:
            prefix = (None, None)

        key_states = project(hidden_states, self.k, key_value_states, prefix[0])
        value_states = project(hidden_states, self.v, key_value_states, prefix[1])

        if self.mode == "store":
            self.stored_key_value_states = (key_states, value_states)

        key_states, value_states = shape(key_states), shape(value_states)

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                # rather than use the relative attention bias, we instead append 0 as the bias to
                # prevent the model struggling to make use of the prefixes.
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                if self.mode == "apply":
                    position_bias = self.compute_bias(seq_length, key_length)[
                        :, :, -seq_length:, :
                    ]
                else:
                    position_bias = self.compute_bias(seq_length, key_length)
            if prefix[0] is not None:
                    position_bias = torch.cat([
                        torch.zeros((1, self.n_heads, seq_length, prefix[0].size(1)), device=scores.device, dtype=scores.dtype), position_bias
                    ], dim=-1)

            if mask is not None:
                if self.mode == "apply":
                    mask = F.pad(mask, value=-0.0, pad=(self.num_prefix_tokens, 0))
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output,) + (None,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    def setup_prefix(self, shared):
        self.prefix_emb = shared["prefix_emb"]
        self.prefix_mlp = nn.Sequential(
            shared["prefix_linear"],
            nn.Tanh(),
            nn.Linear(shared["prefix_linear"].out_features, self.inner_dim * 2),
        )

    def get_prefix(self, bs):
        prefix = self.prefix_mlp(self.prefix_emb.weight)
        batch_prefix = prefix.unsqueeze(dim=0).expand(bs, -1, -1)
        key_prefix, value_prefix = batch_prefix.chunk(dim=-1, chunks=2)
        return key_prefix, value_prefix

    def set_mode(self, mode):
        self.mode = mode
        if self.mode == "store":
            self.stored_key_value_states = None


# TODO: set the parameters via config
def modify_with_mam(transformer: PreTrainedModel, adapter_size: int, num_prefix_tokens: int, prefix_reparam_hidden_size: int):
    # prefix setup
    hidden_size = prefix_reparam_hidden_size
    num_prefix_tokens = num_prefix_tokens
    adapter_size = adapter_size
    shared = {
        "prefix_emb": nn.Embedding(num_prefix_tokens, transformer.config.d_model),
        "prefix_linear": nn.Linear(transformer.config.d_model, hidden_size),
    }
    # attention modules become prefix, ff become adapter
    for _, module in dict(transformer.named_modules()).items():
        if isinstance(module, T5LayerCrossAttention):
            module.EncDecAttention = T5AttentionPrefixTuning(module.EncDecAttention, num_prefix_tokens, hidden_size, shared)
        elif isinstance(module, T5LayerSelfAttention):
            module.SelfAttention = T5AttentionPrefixTuning(module.SelfAttention, num_prefix_tokens, hidden_size, shared)
        if isinstance(module, T5Block):
            module.layer[-1] = T5LayerFFWithAdapter(module.layer[-1], adapter_size, transformer.config.d_model)
    return transformer


@Model.register("mam_seq2seq")
class MaMSeq2Seq(BasicSeq2Seq):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        # settings from "toward unified view of parameter efficient learning"
        # https://arxiv.org/abs/2110.04366
        self.transformer = modify_with_mam(
            self.transformer,
            512,
            30,
            512
        )
        # only train lora parameters
        for name, param in self.transformer.named_parameters():
            # 'prefix_' to get the prefix reparameterisation params
            if "adapter" in name or 'prefix_' in name or 'layer_norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
