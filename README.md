# Transformer WMT14 英德翻译复现

这是一个使用PyTorch从基础模块实现Transformer，并在WMT14 English–German平行语料上完成训练、推理和评估的复现项目。其中自行实现了缩放点积注意力、多头注意力、编码器、解码器、位置编码、标签平滑、Noam学习率、动态token batching、断点恢复、beam search和checkpoint averaging。
本项目最终模型能够生成与源句相关的德语翻译，但测试集得分仍低于原论文。

## 最终结果

最终模型为Transformer Base，并对epoch 14–18的五个模型checkpoint进行参数平均。

| 项目 | 结果 |
| --- | ---: |
| 优化器更新步数 | 100000 |
| 验证集 NLL | 2.3504 |
| 验证集困惑度 | 10.49 |
| 验证集 token accuracy | 56.63% |
| WMT14 `test_filtered` 句数 | 2737 |
| SacreBLEU | **16.4** |
| BP / 长度比 | 1.000 / 1.015 |

SacreBLEU 详细结果：

```text
BLEU = 16.4 48.7/21.6/11.3/6.1
BP = 1.000
ratio = 1.015
hyp_len = 58444
ref_len = 57579
nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
```

原论文报告的Transformer Base结果为27.3 BLEU，但原论文使用旧式tokenized BLEU，本项目报告的是detokenized SacreBLEU。两者不能直接等同；但即使考虑评估方式差异，本项目与论文结果之间仍存在明显差距。

## 实现内容

- 缩放点积注意力与多头注意力
- 正弦/余弦位置编码
- Encoder–Decoder Transformer
- masked self-attention与padding mask
- 共享词表和SentencePiece BPE
- label smoothing cross entropy
- Adam优化器与Noam学习率调度
- 按token数动态组成batch
- 长度池排序与缓冲区打乱
- BF16自动混合精度
- 梯度累加
- checkpoint保存和加载
- epoch中途恢复所需的batch offset
- greedy decoding与beam search
- checkpoint参数平均
- 验证集NLL、困惑度、token accuracy

## 项目结构

```text
.
├── attention.py             #缩放点积注意力与多头注意力
├── encoder.py               #Transformer Encoder
├── decoder.py               #Transformer Decoder
├── transformer.py           #完整Encoder–Decoder模型
├── position.py              #位置编码
├── addnorm.py               #残差连接、LayerNorm和dropout
├── feedforwardnet.py        #Position-wise FFN
├── tokenizer.py             #SentencePiece封装
├── data.py                  #数据集、缓冲区打乱、动态batching
├── loss.py                  #标签平滑交叉熵
├── optimize.py              #Adam与Noam学习率
├── checkpoint.py            #保存、恢复和checkpoint averaging
├── inference.py             #greedy decoding与beam search
├── evaluate.py              #NLL与SacreBLEU 评估
├── train.py                 #训练入口
├── config.py                #Tiny、Base、Big配置
├── scripts/
│   ├── prepare_eval_data.py #准备验证集和测试集(gpt5.6生成)
│   ├── shuffle_data.py      #全局打乱平行训练语料(gpt5.6生成)
│   └── train_tokenizer.py   #训练SentencePiece
└── vocab/
    ├── wmt14_en_de_bpe_37k.model
    └── wmt14_en_de_bpe_37k.vocab
```

## 环境

最终服务器环境：

```text
Python        3.11
PyTorch       2.3.1+cu121
SentencePiece 0.2.2
SacreBLEU     2.6.0
GPU           NVIDIA A10, 22.18 GiB
```

除PyTorch外的依赖：

```bash
python -m pip install sentencepiece==0.2.2 sacrebleu==2.6.0
```

PyTorch应根据本机CUDA环境从官方渠道安装。

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
- `valid.*`：newstest2013，共3000个句对；
- `test_filtered.*`：常用的WMT14 En–De 2737句测试集；
- `test_full.*`：恢复后的3003句完整版本；
- `train.shuffled.*`：对齐句对经过一次全局打乱后的训练语料。

原始 WMT14 数据需要由使用者自行取得。`scripts/prepare_eval_data.py`假定原始开发集与测试集已解压到 `~/datasets/wmt14/extracted/`下的相应目录。

## 全局打乱训练语料

原始训练文件由多个语料顺序拼接。只进行有限缓冲区打乱时，训练日志随语料切换表现出有规律的明显波动，并且训练效果极差，因此训练前需要先对英德句对做同一全局排列。

## 分词器

默认分词器位于：

```text
vocab/wmt14_en_de_bpe_37k.model
```

配置为共享的37000词SentencePiece BPE，特殊token ID为：

```text
PAD = 0
BOS = 1
EOS = 2
UNK = 3
```

`scripts/train_tokenizer.py`可用于重新训练分词器。训练完成后，应将生成的`.model` 和 `.vocab`文件放到项目的`vocab/`目录，并保持上述文件名。

## 模型配置

| 配置 | 层数 | 隐藏维度 | FFN 维度 | 注意力头数 | Dropout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | 2 | 64 | 128 | 4 | 0.1 |
| Base | 6 | 512 | 2048 | 8 | 0.1 |
| Big | 6 | 1024 | 4096 | 16 | 0.3 |

最终实验使用 Base。Big配置已实现并完成服务器二十万步测试，但受单卡显存、代码架构和计算资源限制，没有作为最终结果模型。

Base 训练配置：
```text
max_steps                  100000
warmup_steps               4000
max_sequence_tokens        256
micro-batch token budget   16384
gradient accumulation      3
buffer size                10000
length pool size           1000
label smoothing            0.1
Adam betas                 (0.9, 0.98)
Adam epsilon               1e-9
```

## 训练

在 `train.py` 中选择：

```python
model_size = "base"
```

然后运行：

```bash
python train.py
```
在支持BF16的CUDA GPU上，训练会自动使用BF16 autocast。训练入口会根据`config.py`使用动态token batching和梯度累加。

### 断点恢复

如果存在checkpoints/base_last.pt

`train.py`会自动恢复模型、优化器、epoch、global step、训练历史和epoch内batch offset。`base_last.pt` 用于恢复训练，体积会大于纯模型 checkpoint。

训练过程中生成：

```text
checkpoints/base_last.pt
checkpoints/base_best.pt
checkpoints/history/base_checkpoint_epoch_XXXX_step_XXXXXXXX.pt
```

其中 `history/` 下的文件只保存模型参数和必要数据，用于后续平均。

## Checkpoint averaging

最终实验平均最后五个完整 epoch 的模型 checkpoint：epoch 14–18，对应step 77458、83458、89457、95457和100000。平均后的文件只用于推理和评估，不包含可恢复训练所需的完整优化器状态。


## 实验说明与局限

1. 最终训练运行的前七个epoch使用了按语料拼接的原始训练顺序。发现语料切换引起的周期性指标波动后，从后续epoch开始改用全局打乱语料；最终代码从训练开始就读取`train.shuffled.en/de`。
2. SentencePiece BPE与原论文的数据预处理实现并不完全相同。
3. 最终实验使用单张NVIDIA A10，而原论文使用多GPU和更大的训练资源。
4. 当前beam search没有实现KV cache，影响推理速度，但不影响训练时间。
5. 本项目的SacreBLEU与原论文tokenized BLEU评估协议不同，不能直接对比。
6. 最终16.4 SacreBLEU表明模型已经学习到有效的源句到目标句映射，但尚未达到论文报告的翻译质量。