# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import numpy as np

import paddle
from paddlenlp.data import Stack, Tuple, Pad
from run_glue_deberta import MODEL_CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
             ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.", )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size for prediction.", )
    parser.add_argument(
        "--cls_drop_out",
        default=0.1,
        type=float,
        help="cls drop out.")
    args = parser.parse_args()
    return args


def convert_example(example,
                    tokenizer,
                    max_seq_length=128,
                    pad_to_max_seq_len=False,
                    return_attention_mask=False, ):
    if len(example) == 1:
        encoded_inputs = tokenizer(
            example[0],
            max_seq_len=max_seq_length,
            pad_to_max_seq_len=pad_to_max_seq_len,
            return_attention_mask=return_attention_mask)
    else:
        encoded_inputs = tokenizer(
            example[0],
            text_pair=example[1],
            max_seq_len=max_seq_length,
            pad_to_max_seq_len=pad_to_max_seq_len,
            return_attention_mask=return_attention_mask)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    return input_ids, token_type_ids, attention_mask


def main():
    args = parse_args()
    paddle.set_device(args.device)

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    data = [
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
    label_map = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}

    model = model_class.from_pretrained(args.model_name_or_path,
                                        num_classes=len(label_map.keys()), dropout=args.cls_drop_out)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    examples = []
    for text in data:
        input_ids, segment_ids, attention_mask = convert_example(
            text,
            tokenizer,
            max_seq_length=args.max_seq_length,
            pad_to_max_seq_len=True,
            return_attention_mask=True, )
        examples.append((input_ids, segment_ids, attention_mask))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # token_type_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # attention_mask
    ): fn(samples)

    # Seperates data into some batches.
    batches = [
        examples[idx:idx + args.batch_size]
        for idx in range(0, len(examples), args.batch_size)
    ]

    outputs = []
    results = []
    for batch in batches:
        input_ids, token_type_ids, attention_mask = batchify_fn(batch)
        logits = model(input_ids=paddle.to_tensor(input_ids),
                       token_type_ids=paddle.to_tensor(token_type_ids),
                       attention_mask=paddle.to_tensor(attention_mask))
        probs = paddle.nn.functional.softmax(logits, axis=1)
        pred_res = paddle.argmax(probs, axis=1).numpy()
        pred_res = pred_res.tolist()
        labels = [label_map[i] for i in pred_res]
        outputs.extend(probs)
        results.extend(labels)

    for idx, text in enumerate(data):
        print('Data: {} \n Label: {} \n'.format(text, results[idx]))


if __name__ == "__main__":
    main()
