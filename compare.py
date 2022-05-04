import transformers
from typing import Optional, Tuple, Union
import numpy as np
import paddle
import paddlenlp
import torch
from reprod_log.compare import compute_diff
from reprod_log.utils import paddle2np, torch2np, check_print_diff

from deberta import deberta, deberta_v2, deberta_tokenizer, deberta_v2_tokenizer

import reprod_log


# deberta model
model_name = 'deberta-large'
paddle_model = deberta.DebertaModel.from_pretrained(model_name)
paddle_to = deberta_tokenizer.DebertaTokenizer.from_pretrained(model_name)
torch_model = transformers.models.deberta.DebertaModel.from_pretrained('microsoft/' + model_name)
torch_to = transformers.DebertaTokenizer.from_pretrained('microsoft/' + model_name)


# deberta v2 model
# model_name = 'deberta-v2-xlarge'
# torch_model = transformers.models.deberta_v2.DebertaV2Model.from_pretrained('microsoft/' + model_name)
# paddle_model = deberta_v2.DebertaV2Model.from_pretrained(model_name)
# paddle_to = deberta_v2_tokenizer.DebertaV2Tokenizer.from_pretrained(model_name)
# torch_to = transformers.DebertaV2Tokenizer.from_pretrained('microsoft/' + model_name)

sentences_pairs = [
    [
        "That which binds together Chinese.",
        "This is a shared value among Chinese people."],
    [
        "The actual length of an individual worker's H-2A visa varies depending upon the geographic location of the employer and the nature of the farmwork to be performed.",
        "The location of the employer effects the length of the worker's H-2A visa."
    ],
    [
        "Every man I put down left me empty.",
        "I felt empty after every man I put down."
    ]
]


rl = reprod_log.ReprodLogger()
rdh = reprod_log.ReprodDiffHelper()

for pair in sentences_pairs:
    paddle_emb = paddle_to(pair[0], pair[1], max_seq_len=128, pad_to_max_seq_len=True, return_attention_mask=True)
    torch_emb = torch_to(*pair, max_length=128, padding='max_length')
    # print(paddle_emb)
    # print(torch_emb)
    paddle_input = {k: paddle.to_tensor(v).unsqueeze(0) for k, v in paddle_emb.items()}
    torch_input = {k: torch.tensor(v).unsqueeze(0) for k, v in torch_emb.items()}

    torch_model.eval()
    paddle_model.eval()

    torch_out = torch_model(**torch_input)['last_hidden_state']
    paddle_out, pooled_output = paddle_model(**paddle_input)
    # print(torch_out.shape, torch_out)
    # print(paddle_out.shape, paddle_out)
    diff_dict = compute_diff(torch2np(torch_out), paddle2np(paddle_out))
    for diff_m in ['mean', 'max']:
        print('{} diff: {:.5e}'.format(diff_m, diff_dict['output'][diff_m]))
