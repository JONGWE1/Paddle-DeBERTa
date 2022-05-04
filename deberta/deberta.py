import json
import math
import os
from collections.abc import Sequence
from typing import Optional, Tuple, Union

import paddle
from paddle import nn
from paddlenlp.transformers import PretrainedModel, register_base_model


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

pretrained_init_configuration = {
    "deberta-base": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 0,
        "vocab_size": 50265,
        "pad_token_id": 0,
        "layer_norm_eps": 1e-7,
        "relative_attention": True,
        "max_relative_positions": -1,
        "position_biased_input": False,
        "pos_att_type": ['c2p', 'p2c'],
        "pooler_dropout": 0,
        "pooler_hidden_act": "gelu",
        "pooler_hidden_size": 768
    },
    "deberta-large": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 0,
        "vocab_size": 50265,
        "pad_token_id": 0,
        "layer_norm_eps": 1e-7,
        "relative_attention": True,
        "max_relative_positions": -1,
        "position_biased_input": False,
        "pos_att_type": ['c2p', 'p2c'],
        "pooler_dropout": 0,
        "pooler_hidden_act": "gelu",
        "pooler_hidden_size": 1024
    },
    "deberta-xlarge": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 48,
        "type_vocab_size": 0,
        "vocab_size": 50265,
        "pad_token_id": 0,
        "layer_norm_eps": 1e-7,
        "relative_attention": True,
        "max_relative_positions": -1,
        "position_biased_input": False,
        "pos_att_type": ['c2p', 'p2c'],
        "pooler_dropout": 0,
        "pooler_hidden_act": "gelu",
        "pooler_hidden_size": 1024
    },
}

DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
    "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/config.json",
    "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/config.json",
    "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/config.json",
    "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/config.json",
    "microsoft/deberta-xlarge-mnli": "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/config.json",
}


class DebertaConfig:
    r"""
    This is the configuration class to store the configuration of a [`DebertaModel`] or a [`TFDebertaModel`]. It is
    used to instantiate a DeBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DeBERTa
    [microsoft/deberta-base](https://huggingface.co/microsoft/deberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the DeBERTa model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DebertaModel`] or [`TFDebertaModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
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
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`DebertaModel`] or [`TFDebertaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        relative_attention (`bool`, *optional*, defaults to `False`):
            Whether use relative position encoding.
        max_relative_positions (`int`, *optional*, defaults to 1):
            The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
            as `max_position_embeddings`.
        pad_token_id (`int`, *optional*, defaults to 0):
            The value used to pad input_ids.
        position_biased_input (`bool`, *optional*, defaults to `True`):
            Whether add absolute position embedding to content embedding.
        pos_att_type (`List[str]`, *optional*):
            The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
            `["p2c", "c2p"]`.
        layer_norm_eps (`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
    """
    model_type = "deberta"

    def __init__(
            self,
            vocab_size=50265,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=0,
            initializer_range=0.02,
            layer_norm_eps=1e-7,
            relative_attention=False,
            max_relative_positions=-1,
            pad_token_id=0,
            position_biased_input=True,
            pos_att_type=['c2p', 'p2c'],
            pooler_dropout=0,
            pooler_hidden_act="gelu",
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input

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
        output = paddle.where(mask.astype(paddle.bool).expand_as(input), input, paddle.ones_like(input) * float('-inf'))
        output = nn.functional.softmax(output, dim)
        output = paddle.where(mask.astype(paddle.bool).expand_as(output), output, paddle.zeros_like(output))
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
        """
        Call the module

        Args:
            x (`paddle.tensor`): The input tensor to apply dropout
        """
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


class DebertaLayerNorm(nn.Layer):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, size, eps=1e-12):
        super().__init__()
        self.weight = paddle.create_parameter([size], dtype='float32')
        self.bias = paddle.create_parameter([size], dtype='float32', is_bias=True)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_type = hidden_states.dtype
        hidden_states = hidden_states.astype(paddle.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) / paddle.sqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.astype(input_type)
        y = self.weight * hidden_states + self.bias
        return y


class DebertaSelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaSelfOutput(config)
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


class DebertaIntermediate(nn.Layer):
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


class DebertaOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.attention = DebertaAttention(config)
        self.intermediate = DebertaIntermediate(config)
        self.output = DebertaOutput(config)

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


class DebertaEncoder(nn.Layer):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.LayerList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
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
            relative_pos = build_relative_position(q, hidden_states.shape[-2])
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
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(
                hidden_states=next_kv,
                attention_mask=attention_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
            )

            if output_attentions:
                hidden_states, att_m = hidden_states

            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


def build_relative_position(query_size, key_size):
    """
    Build relative position according to the query and key

    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size), The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k`

    Args:
        query_size (int): the length of query
        key_size (int): the length of key

    Return:
        :obj:`paddle.Tensor`: A tensor with shape [1, query_size, key_size]

    """
    q_ids = paddle.arange(query_size, dtype=paddle.int64)
    k_ids = paddle.arange(key_size, dtype=paddle.int64)
    rel_pos_ids = q_ids[:, None] - k_ids.reshape((1, -1)).tile((query_size, 1))
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.shape[0], query_layer.shape[1], query_layer.shape[2], relative_pos.shape[-1]])


def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.shape[0], query_layer.shape[1], key_layer.shape[-2], key_layer.shape[-2]])


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
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.in_proj = nn.Linear(config.hidden_size, self.all_head_size * 3, bias_attr=False)
        self.q_bias = paddle.create_parameter([self.all_head_size], dtype=paddle.float32, is_bias=True)
        self.v_bias = paddle.create_parameter([self.all_head_size], dtype=paddle.float32, is_bias=True)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []

        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)

        if self.talking_head:
            self.head_logits_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias_attr=False)
            self.head_weights_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias_attr=False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)
            if "c2p" in self.pos_att_type:
                self.pos_proj = nn.Linear(config.hidden_size, self.all_head_size, bias_attr=False)
            if "p2c" in self.pos_att_type:
                self.pos_q_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = tuple(x.shape[:-1], ) + (self.num_attention_heads, -1)
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))

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
            hidden_states (`paddle.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`paddle.ByteTensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            output_attentions (`bool`, optional):
                Whether return the attention matrix.

            query_states (`paddle.FloatTensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`paddle.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`paddle.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        if query_states is None:
            qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, axis=-1)
        else:

            def linear(w, b, x):
                if b is not None:
                    return paddle.matmul(x, w.t()) + b.t()
                else:
                    return paddle.matmul(x, w.t())  # + b.t()

            ws = self.in_proj.weight.chunk(self.num_attention_heads * 3, axis=0)
            qkvw = [paddle.concat([ws[i * 3 + k] for i in range(self.num_attention_heads)], axis=0) for k in range(3)]
            qkvb = [None] * 3

            q = linear(qkvw[0], qkvb[0], query_states)
            k, v = [linear(qkvw[i], qkvb[i], hidden_states) for i in range(1, 3)]
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q, k, v]]

        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(query_layer.shape[-1] * scale_factor)
        query_layer = query_layer / scale
        attention_scores = paddle.matmul(query_layer, key_layer.transpose((0, 1, 3, 2)))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        # bxhxlxd
        if self.talking_head:
            attention_scores = self.head_logits_proj(attention_scores.transpose((0, 2, 3, 1))).transpose((0, 3, 1, 2))

        attention_probs = XSoftmax(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        if self.talking_head:
            attention_probs = self.head_weights_proj(attention_probs.transpose((0, 2, 3, 1))).transpose(0, 3, 1, 2)

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = tuple(context_layer.shape[:-2], ) + (-1,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.shape[-2]
            relative_pos = build_relative_position(q, key_layer.shape[-2])
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = min(max(query_layer.shape[-2], key_layer.shape[-2]), self.max_relative_positions)
        relative_pos = relative_pos.astype(paddle.int64)
        rel_embeddings = rel_embeddings[
                         self.max_relative_positions - att_span: self.max_relative_positions + att_span, :
                         ].unsqueeze(0)

        score = 0

        # content->position
        if "c2p" in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            c2p_att = paddle.matmul(query_layer, pos_key_layer.transpose((0, 1, 3, 2)))
            c2p_pos = paddle.clip(relative_pos + att_span, 0, att_span * 2 - 1)
            b, c, d, _ = tuple(c2p_att.shape)

            c2p_att = paddle.index_sample(
                c2p_att.flatten(0, 2), c2p_dynamic_expand(c2p_pos, query_layer, relative_pos).flatten(0, 2)
            ).reshape((b, c, d, -1))
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            pos_query_layer /= math.sqrt(pos_query_layer.shape[-1] * scale_factor)
            if query_layer.shape[-2] != key_layer.shape[-2]:
                r_pos = build_relative_position(key_layer.shape[-2], key_layer.shape[-2])
            else:
                r_pos = relative_pos
            p2c_pos = paddle.clip(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = paddle.matmul(key_layer, pos_query_layer.transpose((0, 1, 3, 2)))
            b, c, d, _ = tuple(p2c_att.shape)

            p2c_att = paddle.index_sample(
                p2c_att.flatten(0, 2), index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer).flatten(0, 2)
            ).reshape((b, c, d, -1))

            p2c_att = paddle.transpose(p2c_att, (0, 1, 3, 2))
            if query_layer.shape[-2] != key_layer.shape[-2]:
                b, c, d, _ = tuple(p2c_att.shape)

                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                p2c_att = paddle.index_sample(
                    p2c_att.flatten(0, 2), index=pos_dynamic_expand(pos_index, p2c_att, key_layer).flatten(0, 2)
                ).reshape((b, c, d, -1))

            score += p2c_att

        return score


class DebertaEmbeddings(nn.Layer):
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
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
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


class DebertaPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained Deberta models. It provides Deberta related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.

    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "deberta-base": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 0,
            "vocab_size": 50265,
            "pad_token_id": 0,
            "layer_norm_eps": 1e-7,
            "relative_attention": True,
            "max_relative_positions": -1,
            "position_biased_input": False,
            "pos_att_type": ['c2p', 'p2c'],
            "pooler_dropout": 0,
            "pooler_hidden_act": "gelu",
            "pooler_hidden_size": 768
        },
        "deberta-large": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 0,
            "vocab_size": 50265,
            "pad_token_id": 0,
            "layer_norm_eps": 1e-7,
            "relative_attention": True,
            "max_relative_positions": -1,
            "position_biased_input": False,
            "pos_att_type": ['c2p', 'p2c'],
            "pooler_dropout": 0,
            "pooler_hidden_act": "gelu",
            "pooler_hidden_size": 1024
        },
        "deberta-xlarge": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "type_vocab_size": 0,
            "vocab_size": 50265,
            "pad_token_id": 0,
            "layer_norm_eps": 1e-7,
            "relative_attention": True,
            "max_relative_positions": -1,
            "position_biased_input": False,
            "pos_att_type": ['c2p', 'p2c'],
            "pooler_dropout": 0,
            "pooler_hidden_act": "gelu",
            "pooler_hidden_size": 1024
        },
        "deberta-base-mnli": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 0,
            "vocab_size": 50265,
            "pad_token_id": 0,
            "layer_norm_eps": 1e-7,
            "relative_attention": True,
            "max_relative_positions": -1,
            "position_biased_input": False,
            "pos_att_type": ['c2p', 'p2c'],
            "pooler_dropout": 0,
            "pooler_hidden_act": "gelu",
            "pooler_hidden_size": 768
        },
        "deberta-large-mnli": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "type_vocab_size": 0,
            "vocab_size": 50265,
            "pad_token_id": 0,
            "layer_norm_eps": 1e-7,
            "relative_attention": True,
            "max_relative_positions": -1,
            "position_biased_input": False,
            "pos_att_type": ['c2p', 'p2c'],
            "pooler_dropout": 0,
            "pooler_hidden_act": "gelu",
            "pooler_hidden_size": 1024
        },
        "deberta-xlarge-mnli": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "num_attention_heads": 16,
            "num_hidden_layers": 48,
            "type_vocab_size": 0,
            "vocab_size": 50265,
            "pad_token_id": 0,
            "layer_norm_eps": 1e-7,
            "relative_attention": True,
            "max_relative_positions": -1,
            "position_biased_input": False,
            "pos_att_type": ['c2p', 'p2c'],
            "pooler_dropout": 0,
            "pooler_hidden_act": "gelu",
            "pooler_hidden_size": 1024
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "deberta-base": "https://bj.bcebos.com/v1/ai-studio-online/e94fcd167945465f8a58364afaeb029b5fccaac4ab5e4181bd3bf61bd8c50167?responseContentDisposition=attachment%3B%20filename%3Ddeberta-base.pdparams",
            "deberta-large": "https://bj.bcebos.com/v1/ai-studio-online/b3ff36149da94520b34668a5b99f662918531e2f46f441ef987b9c2fa4a6b2ac?responseContentDisposition=attachment%3B%20filename%3Ddeberta-large.pdparams",
            "deberta-xlarge": "https://bj.bcebos.com/v1/ai-studio-online/4339b7346e8c4a089c07bcb122eebfbbecc3fe9401974210a06228c0926dafe1?responseContentDisposition=attachment%3B%20filename%3Ddeberta-xlarge.pdparams",
            "deberta-base-mnli": "https://bj.bcebos.com/v1/ai-studio-online/5ca8a24098e944888f25f8195cd333b42e1a0ccaeaf049fe87abc3571e6ed2fc?responseContentDisposition=attachment%3B%20filename%3Ddeberta-base-mnli.pdparams",
            "deberta-large-mnli": "https://bj.bcebos.com/v1/ai-studio-online/9b538f64053145879561c3f946376e2abc7d1409eb284987ad12e6fcb46b2768?responseContentDisposition=attachment%3B%20filename%3Ddeberta-large-mnli.pdparams",
            "deberta-xlarge-mnli": "https://bj.bcebos.com/v1/ai-studio-online/b3097e9fe7c240eba922588de3aedcde4c78608acf5b4a6586d1ea59099b4933?responseContentDisposition=attachment%3B%20filename%3Ddeberta-xlarge-mnli.pdparams",
        }  # note: the -mnli models is not fine-tuned
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
class DebertaModel(DebertaPretrainedModel):
    def __init__(self,
                 vocab_size=50265,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
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
                 pooler_hidden_size=768):
        super(DebertaModel, self).__init__()
        _config = DebertaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
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
            pooler_hidden_size=pooler_hidden_size)
        self.initializer_range = initializer_range

        self.embeddings = DebertaEmbeddings(_config)
        self.encoder = DebertaEncoder(_config)
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
            attention_mask,
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

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2):]

        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class DebertaForSequenceClassification(DebertaPretrainedModel):
    def __init__(self, deberta, num_classes=3, dropout=None, stop_gradient=None, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.stop_gradient = stop_gradient
        self.pretrained = pretrained

        self.deberta = deberta
        output_dim = self.deberta._config.hidden_size

        self.classifier = nn.Linear(output_dim, self.num_classes)
        drop_out = self.deberta._config.hidden_dropout_prob if dropout is None else dropout
        self.dropout = StableDropout(drop_out)

        self.apply(self._init_weights)

    @paddle.jit.to_static
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

