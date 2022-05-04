# Paddle-DeBERTa

本项目基于PaddlePaddle实现DeBERTa和DeBERTa-v2模型

## 1  简介

1、DeBERTa论文：[ **DeBERTa**: **D**ecoding-**e**nhanced **BERT** with Disentangled **A**ttention ](https://arxiv.org/abs/2006.03654)

2、官方repository：https://github.com/microsoft/DeBERTa

3、官方repository指示预训练模型从Hugging Face下载：https://huggingface.co/microsoft/deberta-large

4、安装Transformers库以快速获取基准结果，具体参见Hugging Face官方repository：https://github.com/huggingface/transformers

5、复现在Baidu AI Studio平台进行。

## 环境依赖

通过以下命令安装对应依赖

```shell
pip install -r requirements.txt
```

## 数据集

复现主要使用了MNLI数据集，本次复现要求为：

1、DeBERTa-large 模型在 MNLI 验证集上 Acc =91.1 / 91.1

2、DeBERTa-v2-xlarge 模型在 MNLI 验证集上 Acc =91.7/91.6



PaddleNLP库提供了数据集的快速读取的API，PaddleNLP官方文档中给出了使用说明，参考链接：https://paddlenlp.readthedocs.io/zh/latest/data_prepare/overview.html

使用以下代码便可简单加载MNLI数据集：

```python
from paddlenlp.datasets import load_dataset

train_ds = load_dataset('glue', 'mnli', splits="train")
dev_ds_matched, dev_ds_mismatched = load_dataset('glue', 'mnli', splits=["dev_matched", "dev_mismatched"])
```

加载后的处理流程在上述参考链接中已经给出，这里不再赘述。

## 权重转换

从Hugging Face下载官方pytorch模型，并转换为Paddle模型，转换教程请参考PaddlePaddle提供的“论文复现赛指南-NLP方向”：https://github.com/PaddlePaddle/models/blob/release/2.2/docs/lwfx/ArticleReproduction_NLP.md

官方模型：https://huggingface.co/microsoft/deberta-large

​				   https://huggingface.co/microsoft/deberta-v2-xlarge

加载模型：

```python
from deberta import deberta, deberta_v2, deberta_tokenizer, deberta_v2_tokenizer

# deberta model
model_name = 'deberta-large'
paddle_model = deberta.DebertaModel.from_pretrained(model_name)
paddle_to = deberta_tokenizer.DebertaTokenizer.from_pretrained(model_name)

# deberta v2 model
model_name = 'deberta-v2-xlarge'
paddle_model = deberta_v2.DebertaV2Model.from_pretrained(model_name)
paddle_to = deberta_v2_tokenizer.DebertaV2Tokenizer.from_pretrained(model_name)
```

运行`python compare.py`，可对比官方pytorch模型和转换后的模型精度情况。

## 快速开始

### 训练

训练+评估的代码参考自https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/bert/run_glue.py

对于单机单卡或者单机多卡的启动脚本及具体参数释义可以参考：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert

官方训练参数设置请见：https://github.com/microsoft/DeBERTa/blob/master/experiments/glue/mnli.sh

对于单机单卡，启动脚本示例如下所示：

```shell
python run_glue_deberta.py \
    --task_name 'mnli' \
    --model_type 'deberta' \
    --model_name_or_path 'deberta-large' \
    --max_seq_length 128 \
    --learning_rate 3e-6 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --train_batch_size 64 \
    --val_batch_size 128 \
    --num_train_epochs 2 \
    --logging_steps 50 \
    --save_steps 1000 \
    --device 'gpu' \
    --warmup_proportion 0.1 \
    --output_dir 'deberta-large-mnli/' \
    --fp16
```

对于单机多卡（示例中为4卡训练），启动脚本示例如下所示：

```shell
python -m paddle.distributed.launch --gpus "0,1,2,3" run_glue_deberta.py \
    --task_name 'mnli' \
    --model_type 'deberta' \
    --model_name_or_path 'deberta-large' \
    --max_seq_length 128 \
    --learning_rate 3e-6 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --train_batch_size 64 \
    --val_batch_size 128 \
    --num_train_epochs 2 \
    --logging_steps 50 \
    --save_steps 1000 \
    --device 'gpu' \
    --warmup_proportion 0.1 \
    --output_dir 'deberta-large-mnli/' \
    --fp16
```

其中参数释义如下：
- `task_name` 指示了具体需要fine-tune的任务。
- `model_type` 指示了模型类型，使用DeBERTa模型时设置为'deberta'即可。
- `model_name_or_path` 指示了某种特定配置的模型，对应有其预训练模型和预训练时使用的 tokenizer。
- `max_seq_length` 表示每个输入句子tokenize后的最大长度，长的序列将被截断，短的序列将被填充。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `max_steps` 表示最大训练步数。若训练`num_train_epochs`轮包含的训练步数大于该值，则达到`max_steps`后就提前结束。
- `train_batch_size` 表示训练时每次迭代**每张卡**上的样本数目。
- `val_batch_size` 表示验证时每次迭代**每张卡**上的样本数目。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `adam_epsilon` 表示AdamW优化器中使用的epsilon值。
- `warmup_steps` 表示动态学习率热启的step数。
- `input_dir` 表示输入数据的目录，该目录下所有文件名中包含training的文件将被作为训练数据。
- `output_dir` 表示模型的保存目录。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `device` 表示训练使用的设备, 'gpu'表示使用GPU, 'xpu'表示使用百度昆仑卡, 'cpu'表示使用CPU。
- `fp16` 指示是否启用自动混合精度训练。

Fine-tuning过程将按照 `logging_steps` 和 `save_steps` 的设置打印日志，eval后保存当前最优模型：

```
...
...
global step 2750/12000, epoch: 0, batch: 2750, rank_id: 0, loss: 0.099526, lr: 0.0000025227, speed: 1.3251 step/s
global step 2800/12000, epoch: 0, batch: 2800, rank_id: 0, loss: 0.171943, lr: 0.0000025091, speed: 1.3334 step/s
global step 2850/12000, epoch: 0, batch: 2850, rank_id: 0, loss: 0.153312, lr: 0.0000024955, speed: 1.3483 step/s
global step 2900/12000, epoch: 0, batch: 2900, rank_id: 0, loss: 0.132467, lr: 0.0000024818, speed: 1.3287 step/s
global step 2950/12000, epoch: 0, batch: 2950, rank_id: 0, loss: 0.212041, lr: 0.0000024682, speed: 1.3316 step/s
global step 3000/12000, epoch: 0, batch: 3000, rank_id: 0, loss: 0.140027, lr: 0.0000024545, speed: 1.3385 step/s
eval loss: 0.17843, acc: 0.912, eval loss: 0.32636, acc: 0.909, eval done total : 264.44744062423706 s
[EVAL] The model with matched dev ACC (0.912) and mismatched dev ACC (0.910) was saved at iter 2000.
...
...
```



注意，对于本项目，复现的训练epoch设置不超过官方数值。复现结果展示如下，这里提供了对应复现模型的下载链接。

|                   | 原模型精度 | 复现精度  | 模型地址                                                     | log                                                          |
| ----------------- | ---------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DeBERTa-large     | 91.1/91.1  | 91.2/91.1 | [deberta-large](https://bj.bcebos.com/v1/ai-studio-online/5109e4dabcff4a6c825b047bd06a838e719aacafb09140d68264cfa6e996d815?responseContentDisposition=attachment%3B%20filename%3Ddeberta-large-mnli.zip) | [deberta-large](https://github.com/JONGWE1/Paddle-DeBERTa/blob/main/log/deberta-large.log) |
| DeBERTa-v2-xlarge | 91.7/91.6  | 91.7/91.6 | [deberta-v2-xlarge](https://bj.bcebos.com/v1/ai-studio-online/580d558d67b34c958c2a17427789746ed4f37cbd3bf44e8f984855b23d87466a?responseContentDisposition=attachment%3B%20filename%3Ddeberta-v2-xlarge-mnli.zip) | [deberta-v2-xlarge](https://github.com/JONGWE1/Paddle-DeBERTa/blob/main/log/deberta-v2-xlarge.log) |

### 预测

预测代码参考https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/bert/predict.py

运行以下命令进行预测：

```shell
unzip deberta-large-mnli.zip
python predict.py --model_type deberta --model_name_or_path deberta-large-mnli/mnli_ft_model_best --device cpu
```

```shell
unzip deberta-v2-xlarge-mnli.zip
python predict.py --model_type deberta-v2 --model_name_or_path deberta-v2-xlarge-mnli/mnli_ft_model_best --device cpu
```

# Reference

```bibtex
@inproceedings{
he2021deberta,
title={DEBERTA: DECODING-ENHANCED BERT WITH DISENTANGLED ATTENTION},
author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=XPZIaotutsD}
}
```
