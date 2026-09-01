# Transformer WMT14 英德翻译复现

这是一个使用 PyTorch 从基础模块实现 Transformer，并在 WMT14 English–German 平行语料上完成训练、推理和评估的复现项目。项目自行实现了注意力、编码器、解码器、位置编码、标签平滑、Noam 学习率、动态 token batching、断点恢复、beam search 和 checkpoint averaging。

最终的 Transformer Base 在 2737 句 `test_filtered` 测试集上取得 **25.05 SacreBLEU**。这一结果来自最后五个模型 checkpoint 的参数平均；同一训练中验证集 NLL 最低的单模型取得 25.03 SacreBLEU。

## 最终结果

| 项目 | 最佳单模型 | 最后五个 checkpoint 平均 |
| --- | ---: | ---: |
| 优化器更新步数 | 100000 | 78005–100000 |
| 测试集句数 | 2737 | 2737 |
| SacreBLEU | **25.03** | **25.05** |
| 1/2/3/4-gram precision | 56.3/30.7/18.9/12.0 | 56.3/30.7/18.9/12.1 |
| BP | 1.000 | 1.000 |
| 输出/参考长度比 | 1.032 | 1.033 |

100000 步模型在 3000 句验证集上的结果：

| 指标 | 结果 |
| --- | ---: |
| NLL | 1.6884 |
| 困惑度 | 5.41 |
| token accuracy | 66.39% |

最终平均模型的完整 SacreBLEU 输出：

```text
BLEU = 25.05 56.3/30.7/18.9/12.1
BP = 1.000
ratio = 1.033
hyp_len = 59479
ref_len = 57579
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
```

评估配置固定为：

```text
beam_size       4
length penalty  0.6
max_new_tokens  256
tokenizer       SacreBLEU 13a
```

原论文报告的 Transformer Base 结果为 27.3 BLEU，但原论文采用旧式 tokenized BLEU，本项目报告 detokenized SacreBLEU。由于数据清洗、子词模型和评估协议并不完全相同，两者不能直接作严格数值比较。

## 收敛过程

以下测试数据均使用完整的 2737 句测试集和同一 beam-search 配置：

| checkpoint | global step | 验证集 NLL | SacreBLEU |
| --- | ---: | ---: | ---: |
| epoch 8 | 48003 | 1.7643 | 24.07 |
| epoch 12 | 72004 | 1.7138 | 24.50 |
| epoch 15 | 90004 | 1.6955 | 24.61 |
| 最佳单模型 | 100000 | 1.6884 | 25.03 |
| 最后五个 checkpoint 平均 | 78005–100000 | — | **25.05** |

## 关键训练修正

### 全局打乱平行语料

原始训练文件由多个语料顺序拼接。仅依靠有限缓冲区打乱时，训练会长时间集中在单一语料，并在切换处出现周期性指标波动，并严重影响训练效果。最终实验在训练开始前，对 4508785 个对齐句对执行一次全局打乱，并从 step 0 开始读取 `train.shuffled.en/de`。

### 按有效 token 加权梯度累积

动态 batching 会让不同 micro-batch 的有效目标 token 数不同。每个 micro-batch 的 loss 已经是 token 平均值，因此直接平均多个 micro-batch loss 会让小 batch 与大 batch 获得相同权重。

最终实现按以下方式计算一次优化器更新的梯度：

```text
step_loss = sum(micro_loss_i * token_count_i) / sum(token_count_i)
```

对应的核心训练逻辑为：

```python
(loss * token_count).backward()
step_tokens += token_count

for parameter in model.parameters():
    if parameter.grad is not None:
        parameter.grad.div_(step_tokens)
```

全局打乱和 token 加权梯度累积是最终训练相对早期失败实验的两个关键调整。早期实验的完整测试集 SacreBLEU 为 16.45；修正后的最终结果上升至 25.05。

## 实现内容

- 缩放点积注意力与多头注意力
- 正弦/余弦位置编码
- Encoder–Decoder Transformer
- masked self-attention 与 padding mask
- 共享词表和 SentencePiece BPE
- label smoothing cross entropy
- Adam 优化器与 Noam 学习率调度
- 按 token 数动态组成 batch
- 长度池排序与缓冲区打乱
- BF16 自动混合精度
- token 加权梯度累积
- step 级和 epoch 级 checkpoint
- epoch 中途恢复所需的 batch offset
- greedy decoding 与 beam search
- checkpoint 参数平均
- 验证集 NLL、困惑度、token accuracy 和 SacreBLEU

## 项目结构

```text
.
├── attention.py             # 缩放点积注意力与多头注意力
├── encoder.py               # Transformer Encoder
├── decoder.py               # Transformer Decoder
├── transformer.py           # 完整 Encoder–Decoder 模型
├── position.py              # 位置编码
├── addnorm.py               # 残差连接、LayerNorm 和 dropout
├── feedforwardnet.py        # Position-wise FFN
├── tokenizer.py             # SentencePiece 封装
├── data.py                  # 数据集、缓冲区打乱、动态 batching
├── loss.py                  # 标签平滑交叉熵
├── optimize.py              # Adam 与 Noam 学习率
├── checkpoint.py            # 保存、恢复和 checkpoint averaging
├── inference.py             # greedy decoding 与 beam search
├── evaluate.py              # NLL 与 SacreBLEU 评估
├── train.py                 # 训练入口
├── config.py                # Tiny、Base、Big 配置
├── scripts/
│   ├── prepare_eval_data.py # 准备验证集和测试集(gpt5.6sol完成)
│   ├── shuffle_data.py      # 全局打乱平行训练语料(gpt5.6sol完成)
│   └── train_tokenizer.py   # 训练 SentencePiece
└── vocab/
    ├── wmt14_en_de_bpe_37k.model
    └── wmt14_en_de_bpe_37k.vocab
```

## 环境

训练先后在两台 NVIDIA A10 实例上断点续训，并全程使用 BF16 autocast：

```text
Python         3.10 / 3.11
PyTorch        2.3.1+cu118 / 2.3.1+cu121
SentencePiece  0.2.2
SacreBLEU      2.6.0
GPU            NVIDIA A10, 22–24 GiB
```

安装除 PyTorch 外的依赖：

```bash
python -m pip install sentencepiece==0.2.2 sacrebleu==2.6.0
```

PyTorch 应根据本机 CUDA 驱动从官方渠道安装。

## 数据目录

训练和评估入口默认读取：

```text
~/datasets/wmt14/processed/plain/
├── train.en
├── train.de
├── train.shuffled.en
├── train.shuffled.de
├── valid.en
├── valid.de
├── test_filtered.en
├── test_filtered.de
├── test_full.en
└── test_full.de
```

其中：

- `train.shuffled.*`：4508785 个经过同一全局排列的训练句对；
- `valid.*`：newstest2013，共 3000 个句对；
- `test_filtered.*`：本项目最终报告使用的 WMT14 En–De 测试集，共 2737 个句对；
- `test_full.*`：保留的 3003 句版本，不用于最终报告。

原始 WMT14 数据需要由使用者自行取得。`scripts/prepare_eval_data.py` 假定原始开发集与测试集已解压到 `~/datasets/wmt14/extracted/` 下的相应目录。

### 全局打乱脚本

```bash
python scripts/shuffle_data.py \
  --src ~/datasets/wmt14/processed/plain/train.en \
  --tgt ~/datasets/wmt14/processed/plain/train.de \
  --output-src ~/datasets/wmt14/processed/plain/train.shuffled.en \
  --output-tgt ~/datasets/wmt14/processed/plain/train.shuffled.de \
  --seed 42
```
当前 Python 脚本会将全部句对载入内存，处理完整 WMT14 训练集时需要数 GB 可用内存。

## 分词器

默认分词器位于：

```text
vocab/wmt14_en_de_bpe_37k.model
```

它是共享的 37000 词 SentencePiece BPE，特殊 token ID 为：

```text
PAD = 0
BOS = 1
EOS = 2
UNK = 3
```

`scripts/train_tokenizer.py` 可用于重新训练分词器。生成的 `.model` 和 `.vocab` 文件需要放在项目的 `vocab/` 目录，并保持上述文件名。

## 模型与训练配置

| 配置 | 层数 | 隐藏维度 | FFN 维度 | 注意力头数 | Dropout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | 2 | 64 | 128 | 4 | 0.1 |
| Base | 6 | 512 | 2048 | 8 | 0.1 |
| Big | 6 | 1024 | 4096 | 16 | 0.3 |

最终实验使用 Base。Big 配置可运行，但没有用于最终结果。

Base 训练配置：

```text
max_steps                    100000
warmup_steps                 4000
max_sequence_tokens          256
micro-batch token budget     16384
gradient accumulation steps  3
buffer size                  10000
length pool size             1000
label smoothing              0.1
Adam betas                   (0.9, 0.98)
Adam epsilon                 1e-9
random seed                  42
```

## 训练与恢复

在 `train.py` 中选择：

```python
model_size = "base"
```

然后运行：

```bash
python train.py
```

在支持 BF16 的 CUDA GPU 上，训练会自动启用 BF16 autocast。

如果存在：

```text
checkpoints/base_last.pt
```

`train.py` 会自动恢复模型、优化器、epoch、global step、训练历史和 epoch 内 batch offset。训练过程中生成：

```text
checkpoints/base_last.pt
checkpoints/base_best.pt
checkpoints/history/base_checkpoint_epoch_XXXX_step_XXXXXXXX.pt
```

`base_last.pt` 和 `base_best.pt` 包含优化器状态，可用于恢复；`history/` 下的文件只保存模型参数和必要数据，用于最后参数平均。

## Checkpoint averaging

运行：

```bash
python checkpoint.py
```

会按照文件名排序，从 `checkpoints/history/` 选择最后五个 Base checkpoint，并生成：

```text
checkpoints/base_averaged_last_5.pt
```

最终实验实际平均了：

```text
epoch 13, step  78005
epoch 14, step  84005
epoch 15, step  90004
epoch 16, step  96005
epoch 17, step 100000
```

step 100000 在第 17 次数据遍历中途触发 `max_steps`，因此最后一个文件是终止时 checkpoint，而不是完整遍历训练集后的 checkpoint。平均模型只用于推理和评估，不包含恢复训练所需的优化器状态。

## 评估

将最终平均模型放在：

```text
checkpoints/base_averaged_last_5.pt
```

在 `evaluate.py` 中设置：

```python
checkpoint_path = project_directory / "checkpoints" / "base_averaged_last_5.pt"
src_path = data_directory / "test_filtered.en"
ref_path = data_directory / "test_filtered.de"

max_examples = None
method = "beam"
max_new_tokens = 256
beam_size = 4
alpha = 0.6
```

运行完整测试集评估：

```bash
python evaluate.py
```

## 实验说明与局限

1. SentencePiece BPE、语料清洗和训练数据组成与原论文的数据流水线并不完全相同。
2. 本项目使用单张 NVIDIA A10，并通过梯度累积扩大每次优化器更新覆盖的 token 数；原论文使用多 GPU 训练。
3. 当前 beam search 没有实现 KV cache，因此推理速度较慢，但不会改变训练结果或解码语义。
4. 最终报告使用 detokenized SacreBLEU 13a，与原论文的旧式 tokenized BLEU 不能直接严格比较。
5. 最终结果基于本项目固定的 2737 句 `test_filtered` 集；更换数据版本或预处理方式会改变分数。