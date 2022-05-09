import json
import math
import os
from collections.abc import Sequence
from paddlenlp.transformers import PretrainedModel, register_base_model
from typing import Optional, Tuple, Union

import numpy as np
import paddle
from paddle import nn


def gelu_python(x): return x * 0.5 * (1.0 + paddle.erf(x / math.sqrt(2.0)))


def gelu_new(x): return 0.5 * x * (1.0 + paddle.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


def gelu_fast(x): return 0.5 * x * (1.0 + paddle.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def quick_gelu(x): return x * nn.functional.sigmoid(1.702 * x)


def linear_act(x): return x


ACT2FN = {
    "relu": nn.functional.relu,
    "silu": nn.functional.silu,
    "swish": nn.functional.silu,
    "gelu": nn.functional.gelu,
    "tanh": paddle.tanh,
    "gelu_python": gelu_python,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "quick_gelu": quick_gelu,
    "mish": nn.functional.mish,
    "linear": linear_act,
    "sigmoid": paddle.nn.functional.sigmoid,
}

DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/config.json",
    "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/config.json",
    "microsoft/deberta-v2-xlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/config.json",
    "microsoft/deberta-v2-xxlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/config.json",
}


class DebertaV2Config:
    r"""
       This is the configuration class to store the configuration of a [`DebertaV2Model`]. It is used to instantiate a
       DeBERTa-v2 model according to the specified arguments, defining the model architecture. Instantiating a
       configuration with the defaults will yield a similar configuration to that of the DeBERTa
       [microsoft/deberta-v2-xlarge](https://huggingface.co/microsoft/deberta-base) architecture.

       Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
       documentation from [`PretrainedConfig`] for more information.

       Arguments:
           vocab_size (`int`, *optional*, defaults to 128100):
               Vocabulary size of the DeBERTa-v2 model. Defines the number of different tokens that can be represented by
               the `inputs_ids` passed when calling [`DebertaV2Model`].
           hidden_size (`int`, *optional*, defaults to 1536):
               Dimensionality of the encoder layers and the pooler layer.
           num_hidden_layers (`int`, *optional*, defaults to 24):
               Number of hidden layers in the Transformer encoder.
           num_attention_heads (`int`, *optional*, defaults to 24):
               Number of attention heads for each attention layer in the Transformer encoder.
           intermediate_size (`int`, *optional*, defaults to 6144):
               Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
           hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
               The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
               `"relu"`, `"silu"`, `"gelu"`, `"tanh"`, `"gelu_fast"`, `"mish"`, `"linear"`, `"sigmoid"` and `"gelu_new"`
               are supported.
           hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
               The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
           attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
               The dropout ratio for the attention probabilities.
           max_position_embeddings (`int`, *optional*, defaults to 512):
               The maximum sequence length that this model might ever be used with. Typically set this to something large
               just in case (e.g., 512 or 1024 or 2048).
           type_vocab_size (`int`, *optional*, defaults to 0):
               The vocabulary size of the `token_type_ids` passed when calling [`DebertaModel`] or [`TFDebertaModel`].
           initializer_range (`float`, *optional*, defaults to 0.02):
               The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
           layer_norm_eps (`float`, *optional*, defaults to 1e-7):
               The epsilon used by the layer normalization layers.
           relative_attention (`bool`, *optional*, defaults to `True`):
               Whether use relative position encoding.
           max_relative_positions (`int`, *optional*, defaults to -1):
               The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
               as `max_position_embeddings`.
           pad_token_id (`int`, *optional*, defaults to 0):
               The value used to pad input_ids.
           position_biased_input (`bool`, *optional*, defaults to `False`):
               Whether add absolute position embedding to content embedding.
           pos_att_type (`List[str]`, *optional*):
               The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
               `["p2c", "c2p"]`, `["p2c", "c2p"]`.
           layer_norm_eps (`float`, optional, defaults to 1e-12):
               The epsilon used by the layer normalization layers.
       """
    model_type = "deberta-v2"

    def __init__(
            self,
            vocab_size=128100,
            hidden_size=1536,
            num_hidden_layers=24,
            num_attention_heads=24,
            intermediate_size=6144,
            conv_act="gelu",
            conv_kernel_size=3,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_head_size=64,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=0,
            initializer_range=0.02,
            layer_norm_eps=1e-7,
            relative_attention=True,
            max_relative_positions=-1,
            pad_token_id=0,
            position_biased_input=False,
            pos_att_type=['c2p', 'p2c'],
            pooler_dropout=0,
            pooler_hidden_act="gelu",
            norm_rel_ebd="layer_norm",
            position_buckets=256,
            share_att_key=True,
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.conv_act = conv_act
        self.conv_kernel_size = conv_kernel_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_head_size = attention_head_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input
        self.norm_rel_ebd = norm_rel_ebd
        self.position_buckets = position_buckets
        self.share_att_key = share_att_key

        # Backwards compatibility
        if type(pos_att_type) == str:
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]

        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps

        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act

        self.output_attentions = False
        self.output_hidden_states = False
        self.use_return_dict = True


class ContextPooler(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


class XSoftmaxLayer(nn.Layer):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`paddle.tensor`): The input tensor that will apply softmax.
        mask (`paddle.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    """

    def forward(self, input, mask, dim):
        output = paddle.where(
            mask.astype(paddle.bool).expand_as(input),
            input,
            paddle.ones_like(input) * float('-inf')
        )
        output = nn.functional.softmax(output, dim)
        output = paddle.where(
            mask.astype(paddle.bool).expand_as(output),
            output,
            paddle.zeros_like(output)
        )

        return output


XSoftmax = XSoftmaxLayer()


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - paddle.bernoulli((paddle.ones_like(input) * (1 - dropout)))).astype(paddle.bool)

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(paddle.autograd.PyLayer):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return paddle.where(mask, paddle.zeros_like(input), input) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensor()
            return paddle.where(mask, paddle.zeros_like(grad_output), grad_output) * ctx.scale
        else:
            return grad_output


class StableDropout(nn.Layer):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


class DebertaV2SelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Attention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)
        self.config = config

    def forward(
            self,
            hidden_states,
            attention_mask,
            output_attentions=False,
            query_states=None,
            relative_pos=None,
            rel_embeddings=None,
    ):
        self_output = self.self(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return attention_output


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Deberta
class DebertaV2Intermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaV2Output(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Layer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(
            self,
            hidden_states,
            attention_mask,
            query_states=None,
            relative_pos=None,
            rel_embeddings=None,
            output_attentions=False,
    ):
        attention_output = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output


class ConvLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = nn.Conv1D(
            config.hidden_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        conv_out = self.conv(hidden_states.transpose((0, 2, 1)))
        out = conv_out.transpose((0, 2, 1))
        # rmask = (1 - input_mask).bool()
        # rmask = (1 - input_mask).astype(paddle.bool)
        # out.masked_fill_(rmask.unsqueeze(-1).expand(out.shape), 0)
        out = paddle.where(
            input_mask.astype(paddle.bool).unsqueeze(-1).expand_as(out),
            out,
            paddle.zeros_like(out)
        )
        out = ACT2FN[self.conv_act](self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input)

        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.astype(output.dtype)
            output_states = output * input_mask

        return output_states


class DebertaV2Encoder(nn.Layer):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.LayerList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.astype('int32')
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.shape[-2] if query_states is not None else hidden_states.shape[-2]
            relative_pos = build_relative_position(
                q, hidden_states.shape[-2], bucket_size=self.position_buckets, max_position=self.max_relative_positions
            )
        return relative_pos

    def forward(
            self,
            hidden_states,
            attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            query_states=None,
            relative_pos=None,
            return_dict=True,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).astype('int32')
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            output_states = layer_module(
                hidden_states=next_kv,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )
            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = np.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = np.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, np.abs(relative_pos))
    log_pos = np.ceil(np.log(abs_pos / mid) / np.log((max_position - 1) / mid) * (mid - 1)) + mid
    bucket_pos = np.where(abs_pos <= mid, relative_pos, log_pos * sign).astype(np.int)
    return bucket_pos


def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position

    Return:
        `torch.LongTensor`: A tensor with shape [1, query_size, key_size]

    """
    q_ids = np.arange(0, query_size)
    k_ids = np.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0], 1))
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = paddle.to_tensor(rel_pos_ids, dtype='int64')
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


# @paddle.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], relative_pos.shape[-1]])


# @paddle.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.shape[0], query_layer.shape[1], key_layer.shape[-2], key_layer.shape[-2]])


# @paddle.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.shape[:2] + (pos_index.shape[-2], key_layer.shape[-2]))


class DisentangledSelfAttention(nn.Layer):
    """
    Disentangled self-attention module

    Parameters:
        config (:obj:`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            `BertConfig`, for more details, please refer :class:`~transformers.DebertaConfig`

    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias_attr=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias_attr=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias_attr=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                # if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias_attr=False)
                # if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = tuple(x.shape[:-1], ) + (attention_heads, -1)
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3)).reshape((-1, x.shape[1], x.shape[-1]))

    def forward(
            self,
            hidden_states,
            attention_mask,
            output_attentions=False,
            query_states=None,
            relative_pos=None,
            rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`torch.ByteTensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            output_attentions (`bool`, optional):
                Whether return the attention matrix.

            query_states (`torch.FloatTensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = math.sqrt(query_layer.shape[-1] * scale_factor)
        attention_scores = paddle.bmm(query_layer, key_layer.transpose((0, 2, 1))) / scale
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = attention_scores
        attention_scores = attention_scores.reshape(
            (-1, self.num_attention_heads, attention_scores.shape[-2], attention_scores.shape[-1])
        )

        # bsz x height x length x dimension
        attention_probs = XSoftmax(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        context_layer = paddle.bmm(
            attention_probs.reshape((-1, attention_probs.shape[-2], attention_probs.shape[-1])), value_layer
        )
        context_layer = context_layer.reshape(
            (-1, self.num_attention_heads, context_layer.shape[-2], context_layer.shape[-1])
        ).transpose((0, 2, 1, 3))
        new_context_layer_shape = tuple(context_layer.shape[:-2], ) + (-1,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.shape[-2]
            relative_pos = build_relative_position(
                q, key_layer.shape[-2], bucket_size=self.position_buckets, max_position=self.max_relative_positions
            )
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bsz x height x query x key
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.astype(paddle.int64)

        rel_embeddings = rel_embeddings[0: att_span * 2, :].unsqueeze(0)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            ).tile((query_layer.shape[0] // self.num_attention_heads, 1, 1))
            pos_key_layer = self.transpose_for_scores(
                self.key_proj(rel_embeddings), self.num_attention_heads
            ).tile((query_layer.shape[0] // self.num_attention_heads, 1, 1))
        else:
            # if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                ).tile((query_layer.shape[0] // self.num_attention_heads, 1, 1))  # .split(self.all_head_size, dim=-1)
            # if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ).tile((query_layer.shape[0] // self.num_attention_heads, 1, 1))  # .split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            scale = math.sqrt(pos_key_layer.shape[-1] * scale_factor)
            c2p_att = paddle.bmm(query_layer, pos_key_layer.transpose((0, 2, 1)))
            c2p_pos = paddle.clip(relative_pos + att_span, 0, att_span * 2 - 1)
            b, c, d = query_layer.shape[0], query_layer.shape[1], relative_pos.shape[-1]
            c2p_att = paddle.index_sample(
                c2p_att.flatten(0, -2),
                c2p_pos.squeeze(0).expand([b, c, d]).flatten(0, -2)
            ).reshape((b, c, d))

            score += c2p_att / scale

        # position->content
        # if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
        if "p2c" in self.pos_att_type:
            scale = math.sqrt(pos_query_layer.shape[-1] * scale_factor)
            if query_layer.shape[-2] != key_layer.shape[-2]:
                r_pos = build_relative_position(
                    key_layer.shape[-2],
                    key_layer.shape[-2],
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions
                )
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = paddle.clip(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = paddle.bmm(key_layer, pos_query_layer.transpose((0, 2, 1)))
            b, c, d = query_layer.shape[0], key_layer.shape[-2], key_layer.shape[-2]
            p2c_att = paddle.index_sample(
                p2c_att.flatten(0, -2),
                index=p2c_pos.squeeze(0).expand([b, c, d]).flatten(0, -2)
            ).reshape((b, c, d)).transpose((0, 2, 1))
            score += p2c_att / scale

        return score


class DebertaV2Embeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias_attr=False)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", paddle.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.astype(paddle.int64))
        else:
            position_embeddings = paddle.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.astype(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaV2PretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained Deberta models. It provides Deberta related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "deberta-v2-xlarge": {
            "attention_head_size": 64,
            "attention_probs_dropout_prob": 0.1,
            "conv_act": "gelu",
            "conv_kernel_size": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1536,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "layer_norm_eps": 1e-7,
            "max_position_embeddings": 512,
            "max_relative_positions": -1,
            "norm_rel_ebd": "layer_norm",
            "num_attention_heads": 24,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "pooler_dropout": 0,
            "pooler_hidden_act": "gelu",
            "pooler_hidden_size": 1536,
            "pos_att_type": [
                "p2c",
                "c2p"
            ],
            "position_biased_input": False,
            "position_buckets": 256,
            "relative_attention": True,
            "share_att_key": True,
            "type_vocab_size": 0,
            "vocab_size": 128100
        },
        "deberta-v2-xxlarge": {
            "attention_head_size": 64,
            "attention_probs_dropout_prob": 0.1,
            "conv_act": "gelu",
            "conv_kernel_size": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1536,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "layer_norm_eps": 1e-07,
            "max_position_embeddings": 512,
            "max_relative_positions": -1,
            "norm_rel_ebd": "layer_norm",
            "num_attention_heads": 24,
            "num_hidden_layers": 48,
            "pad_token_id": 0,
            "pooler_dropout": 0,
            "pooler_hidden_act": "gelu",
            "pooler_hidden_size": 1536,
            "pos_att_type": [
                "p2c",
                "c2p"
            ],
            "position_biased_input": False,
            "position_buckets": 256,
            "relative_attention": True,
            "share_att_key": True,
            "type_vocab_size": 0,
            "vocab_size": 128100
        },
        "deberta-v2-xlarge-mnli": {
            "attention_head_size": 64,
            "attention_probs_dropout_prob": 0.1,
            "conv_act": "gelu",
            "conv_kernel_size": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1536,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "layer_norm_eps": 1e-7,
            "max_position_embeddings": 512,
            "max_relative_positions": -1,
            "norm_rel_ebd": "layer_norm",
            "num_attention_heads": 24,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "pooler_dropout": 0,
            "pooler_hidden_act": "gelu",
            "pooler_hidden_size": 1536,
            "pos_att_type": [
                "p2c",
                "c2p"
            ],
            "position_biased_input": False,
            "position_buckets": 256,
            "relative_attention": True,
            "share_att_key": True,
            "type_vocab_size": 0,
            "vocab_size": 128100
        },
        "deberta-v2-xxlarge-mnli": {
            "attention_head_size": 64,
            "attention_probs_dropout_prob": 0.1,
            "conv_act": "gelu",
            "conv_kernel_size": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1536,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "layer_norm_eps": 1e-07,
            "max_position_embeddings": 512,
            "max_relative_positions": -1,
            "norm_rel_ebd": "layer_norm",
            "num_attention_heads": 24,
            "num_hidden_layers": 48,
            "pad_token_id": 0,
            "pooler_dropout": 0,
            "pooler_hidden_act": "gelu",
            "pooler_hidden_size": 1536,
            "pos_att_type": [
                "p2c",
                "c2p"
            ],
            "position_biased_input": False,
            "position_buckets": 256,
            "relative_attention": True,
            "share_att_key": True,
            "type_vocab_size": 0,
            "vocab_size": 128100
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "deberta-v2-xlarge": "https://bj.bcebos.com/v1/ai-studio-online/78144a99c78d4607958fedf86dd4a23bde235d018a854611a84010d2311ff842?responseContentDisposition=attachment%3B%20filename%3Ddeberta-v2-xlarge.pdparams",
            "deberta-v2-xxlarge": "https://bj.bcebos.com/v1/ai-studio-online/f3dcefba395545358215946463eaef49cdbdb9b58ab24818aa791e7d733a8ec0?responseContentDisposition=attachment%3B%20filename%3Ddeberta-v2-xxlarge.pdparams",
            "deberta-v2-xlarge-mnli": "https://bj.bcebos.com/v1/ai-studio-online/541fb6d3999047a28651cd58ad9f1a4de9c7f1cd61bd473995cbc3dcf671eaeb?responseContentDisposition=attachment%3B%20filename%3Ddeberta-v2-xlarge-mnli.pdparams",
            "deberta-v2-xxlarge-mnli": "https://bj.bcebos.com/v1/ai-studio-online/cc69595574ff4b9bb605408cfc0b43f4faf80e2529444f9faa538cbd54545bb4?responseContentDisposition=attachment%3B%20filename%3Ddeberta-v2-xxlarge-mnli.pdparams",
        }
    }
    base_model_prefix = "deberta"

    def _init_weights(self, layer):
        """Initialize the weights."""
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.deberta._config.initializer_range,
                    shape=layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.Embedding):
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.deberta._config.initializer_range,
                    shape=layer.weight.shape))
            if layer._padding_idx is not None:
                layer.weight[layer._padding_idx].set_value(
                    paddle.zeros_like(layer.weight[layer._padding_idx]))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))


@register_base_model
class DebertaV2Model(DebertaV2PretrainedModel):
    def __init__(self,
                 vocab_size=128100,
                 hidden_size=1536,
                 num_hidden_layers=24,
                 num_attention_heads=24,
                 intermediate_size=6144,
                 conv_act="gelu",
                 conv_kernel_size=3,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_head_size=64,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=0,
                 initializer_range=0.02,
                 layer_norm_eps=1e-7,
                 relative_attention=True,
                 max_relative_positions=-1,
                 pad_token_id=0,
                 position_biased_input=False,
                 pos_att_type=['c2p', 'p2c'],
                 pooler_dropout=0,
                 pooler_hidden_act="gelu",
                 pooler_hidden_size=1536,
                 norm_rel_ebd="layer_norm",
                 position_buckets=256,
                 share_att_key=True):
        super(DebertaV2Model, self).__init__()
        _config = DebertaV2Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            conv_act=conv_act,
            conv_kernel_size=conv_kernel_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_head_size=attention_head_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            relative_attention=relative_attention,
            max_relative_positions=max_relative_positions,
            pad_token_id=pad_token_id,
            position_biased_input=position_biased_input,
            pos_att_type=pos_att_type,
            pooler_dropout=pooler_dropout,
            pooler_hidden_act=pooler_hidden_act,
            pooler_hidden_size=pooler_hidden_size,
            norm_rel_ebd=norm_rel_ebd,
            position_buckets=position_buckets,
            share_att_key=share_att_key)
        self.initializer_range = initializer_range

        self.embeddings = DebertaV2Embeddings(_config)
        self.encoder = DebertaV2Encoder(_config)
        self.pooler = ContextPooler(_config)
        self.z_steps = 0
        self._config = _config
        self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self._config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self._config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self._config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            query_states = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2):]

        return sequence_output, pooled_output


class DebertaV2ForSequenceClassification(DebertaV2PretrainedModel):
    def __init__(self, deberta, num_classes=3, dropout=None):
        super().__init__()
        self.num_classes = num_classes

        self.deberta = deberta
        output_dim = self.deberta._config.hidden_size

        self.classifier = nn.Linear(output_dim, self.num_classes)
        drop_out = self.deberta._config.hidden_dropout_prob if dropout is None else dropout
        self.dropout = StableDropout(drop_out)

        self.apply(self._init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        _, pooled_output = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class DebertaV2ForTokenClassification(DebertaV2PretrainedModel):
    def __init__(self, deberta, num_classes=3, dropout=None):
        super().__init__()
        self.num_classes = num_classes

        self.deberta = deberta
        output_dim = self.deberta._config.hidden_size

        self.classifier = nn.Linear(output_dim, self.num_classes)
        drop_out = self.deberta._config.hidden_dropout_prob if dropout is None else dropout
        self.dropout = StableDropout(drop_out)

        self.apply(self._init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        sequence_output, _ = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits


class DebertaV2ForQuestionAnswering(DebertaV2PretrainedModel):
    def __init__(self, deberta, dropout=None):
        super().__init__()
        self.deberta = deberta

        output_dim = self.deberta._config.hidden_size
        self.classifier = nn.Linear(output_dim, 2)

        self.apply(self._init_weights)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        sequence_output, _ = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits