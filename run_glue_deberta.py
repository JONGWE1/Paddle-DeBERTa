# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import logging
import os
import random
import time
from collections import deque
from functools import partial

import numpy as np
import shutil
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict

from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

import paddle

from deberta import deberta, deberta_v2, deberta_tokenizer, deberta_v2_tokenizer


FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": paddle.metric.Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": paddle.metric.Accuracy,
    "qnli": paddle.metric.Accuracy,
    "rte": paddle.metric.Accuracy,
}

MODEL_CLASSES = {
    "deberta": (deberta.DebertaForSequenceClassification, deberta_tokenizer.DebertaTokenizer),
    "deberta-v2": (deberta_v2.DebertaV2ForSequenceClassification, deberta_v2_tokenizer.DebertaV2Tokenizer),
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()), )
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
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])), )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--val_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for validation.", )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion"
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")

    parser.add_argument(
        "--cls_drop_out",
        default=0.1,
        type=float,
        help="cls drop out.")

    parser.add_argument(
        '--fp16',
        dest='fp16',
        help='Whether to use amp',
        action='store_true')
    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader):
    model.eval()
    metric.reset()
    for batch in data_loader:
        input_ids, token_type_ids, attention_mask, labels = batch

        logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = loss_fct(logits, labels)
        correct = metric.compute(logits, labels)
        metric.update(correct)
    res = metric.accumulate()
    if isinstance(metric, AccuracyAndF1):
        print(
            "eval loss: {:.5f}, acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}, acc and f1: {:.3f}, "
            .format(
                float(loss.numpy()),
                float(res[0]),
                float(res[1]),
                float(res[2]),
                float(res[3]),
                float(res[4])), end='')
    elif isinstance(metric, Mcc):
        print("eval loss: {:.5f}, mcc: {:.3f}, ".format(float(loss.numpy()), float(res[0])), end='')
    elif isinstance(metric, PearsonAndSpearman):
        print(
            "eval loss: %f, pearson: %s, spearman: %s, pearson and spearman: %s, "
            % (loss.numpy(), res[0], res[1], res[2]))
    else:
        print("eval loss: {:.5f}, acc: {:.3f}, ".format(float(loss.numpy()), float(res)), end='')
    model.train()

    return res


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    pad_to_max_seq_len=True,
                    return_attention_mask=True,
                    is_test=False):
    """convert a glue example into necessary features"""
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['labels']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(example['sentence'],
                            max_seq_len=max_seq_length,
                            pad_to_max_seq_len=pad_to_max_seq_len,
                            return_attention_mask=return_attention_mask)
    else:
        example = tokenizer(
            example['sentence1'],
            text_pair=example['sentence2'],
            max_seq_len=max_seq_length,
            pad_to_max_seq_len=pad_to_max_seq_len,
            return_attention_mask=return_attention_mask)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], example['attention_mask'], label
    else:
        return example['input_ids'], example['token_type_ids'], example['attention_mask']


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        # paddle.distributed.init_parallel_env()
        print("Using {} GPUs for training.".format(paddle.distributed.get_world_size()))
        paddle.distributed.fleet.init(is_collective=True)

    set_seed(args)

    args.task_name = args.task_name.lower()
    metric_class = METRIC_CLASSES[args.task_name]
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    train_ds = load_dataset('glue', args.task_name, splits="train")
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        label_list=train_ds.label_list,
        max_seq_length=args.max_seq_length)
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.train_batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # attention_mask
        Stack(dtype="int64" if train_ds.label_list else "float32")  # label
    ): fn(samples)
    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)
    if args.task_name == "mnli":
        dev_ds_matched, dev_ds_mismatched = load_dataset(
            'glue', args.task_name, splits=["dev_matched", "dev_mismatched"])

        dev_ds_matched = dev_ds_matched.map(trans_func, lazy=True)
        dev_ds_mismatched = dev_ds_mismatched.map(trans_func, lazy=True)
        dev_batch_sampler_matched = paddle.io.DistributedBatchSampler(
            dev_ds_matched, batch_size=args.val_batch_size, shuffle=False)
        dev_data_loader_matched = paddle.io.DataLoader(
            dataset=dev_ds_matched,
            batch_sampler=dev_batch_sampler_matched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
        dev_batch_sampler_mismatched = paddle.io.DistributedBatchSampler(
            dev_ds_mismatched, batch_size=args.val_batch_size, shuffle=False)
        dev_data_loader_mismatched = paddle.io.DataLoader(
            dataset=dev_ds_mismatched,
            batch_sampler=dev_batch_sampler_mismatched,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)
    else:
        dev_ds = load_dataset('glue', args.task_name, splits='dev')
        dev_ds = dev_ds.map(trans_func, lazy=True)
        dev_batch_sampler = paddle.io.DistributedBatchSampler(
            dev_ds, batch_size=args.val_batch_size, shuffle=False)
        dev_data_loader = paddle.io.DataLoader(
            dataset=dev_ds,
            batch_sampler=dev_batch_sampler,
            collate_fn=batchify_fn,
            num_workers=0,
            return_list=True)

    num_classes = 1 if train_ds.label_list == None else len(train_ds.label_list)
    model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=num_classes, dropout=args.cls_drop_out)

    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = int(np.ceil(num_training_steps / len(train_data_loader)))
    else:
        num_training_steps = len(train_data_loader) * args.num_train_epochs
        num_train_epochs = args.num_train_epochs

    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, warmup)
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    loss_fct = paddle.nn.loss.CrossEntropyLoss() if train_ds.label_list else paddle.nn.loss.MSELoss()

    metric = metric_class()

    if paddle.distributed.get_world_size() > 1:
        optimizer = paddle.distributed.fleet.distributed_optimizer(optimizer)  # The return is Fleet object
        model = paddle.distributed.fleet.distributed_model(model)

    # use amp
    if args.fp16:
        print('Using Mixed Precision Training')
        scaler = paddle.amp.GradScaler(init_loss_scaling=2**15)

    save_models = deque()
    keep_checkpoint_max = 1
    best_metric = 0
    global_step = 0
    best_model_iter = 0
    best_res1 = 0
    best_res2 = 0
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            tic_train = time.time()
            global_step += 1
            if global_step > num_training_steps:
                continue

            input_ids, token_type_ids, attention_mask, labels = batch

            with paddle.amp.auto_cast(enable=args.fp16):
                if paddle.distributed.get_world_size() > 1:
                    with model.no_sync():
                        logits = model(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask
                        )
                        loss = loss_fct(logits, labels)
                        if args.fp16:
                            scaler.scale(loss).backward()  # scale the loss
                            # step 2 : fuse + allreduce manually before optimization
                            fused_allreduce_gradients(list(model.parameters()), None)
                            scaler.minimize(optimizer, loss)  # do backward
                        else:
                            loss.backward()
                            # step 2 : fuse + allreduce manually before optimization
                            fused_allreduce_gradients(list(model.parameters()), None)
                            optimizer.step()
                else:
                    logits = model(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask
                    )
                    loss = loss_fct(logits, labels)
                    if args.fp16:
                        scaler.scale(loss).backward()  # scale the loss
                        scaler.minimize(optimizer, loss)  # do backward
                    else:
                        loss.backward()
                        optimizer.step()

            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f s/step"
                    % (global_step, num_training_steps, epoch, global_step,
                       paddle.distributed.get_rank(), loss, optimizer.get_lr(), time.time() - tic_train))
            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                if args.task_name == "mnli":
                    res1 = evaluate(model, loss_fct, metric, dev_data_loader_matched)
                    res2 = evaluate(model, loss_fct, metric, dev_data_loader_mismatched)
                    res_min = min(res1, res2)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                else:
                    evaluate(model, loss_fct, metric, dev_data_loader)
                    print("eval done total : %s s" % (time.time() - tic_eval))
                if paddle.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir, "%s_ft_model_%d" %(args.task_name, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Need better way to get inner model of DataParallel
                    if keep_checkpoint_max > 0:
                        model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        save_models.append(output_dir)
                        if len(save_models) > keep_checkpoint_max:
                            model_to_remove = save_models.popleft()
                            shutil.rmtree(model_to_remove)
                    if res_min > best_metric:
                        output_dir = os.path.join(args.output_dir,
                                                  "%s_ft_model_best" % (args.task_name))
                        model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        best_metric = res_min
                        best_res1 = res1
                        best_res2 = res2
                        best_model_iter = global_step
                    print(
                        '[EVAL] The model with matched dev ACC ({:.3f}) and mismatched dev ACC ({:.3f}) was saved at iter {}.'
                        .format(best_res1, best_res2, best_model_iter))


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_train(args)
