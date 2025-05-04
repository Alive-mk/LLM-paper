# **Attention Is All You Need**

## Abstract

```json
sequence transduction model,recurrent,convolutional,dispense,solely,be superior in,parallelizable,literature
state of the art,
```

- `BLEU`
  `BLEU`（Bilingual Evaluation Understudy）是一种用于评估机器翻译质量的指标，通过比较机器翻译输出和一组参考翻译之间的n-gram重叠来评分。

## Introduction

```
numerous,preclude,inherently,sequential nature,
```

- long short-term memory neural networks
- gated recurrent neural networks
- sequence modeling and transduction
- `RNN`中其中 h_t 由上一个隐藏状态 h_{t-1} 和位置 t 的输入共同决定
- factorization tricks
- conditional computation

## Background

```json
resolution,arbitary,counteract,
```

- transduction model
  `seq2seq` model
- 在传统模型中，把任意两个位置之间的信号关联起来所需要的运算量依赖着两者之间的距离，而transformer将这个计算削减至常量
- end-end memory networks

## Model architecture

```json
point-wise,identical,residual,implemented,facilitate,attend to(关注),compatibility function,simultaneously,
pack together,additive,multiplicative,magnitude,gradient,yield,concat,inhibt,mimics,leftward,to this end,
sinusoid,geometric procession,sinusoidal,extrapolate,
```

- 编码器把一个输入序列映射成连续表示序列Z，然后解码器会根据Z按照一步产生一个元素的方式产生一个输出序列。解码器采用的是**自回归**的方式，在生成下一个符号的的时候会把前面已经生成的符号作为额外的输入。

- 堆叠的自注意力机制和逐点的全连接层。

### Encoder and decoder stacks

- masking
- shifted right

<img src="pictures\image-20250426224402795.png" alt="image-20250426224402795" style="zoom:67%;" />

#### Encoder

- 6 identical layers, 2 sub-layers, d_model = 512

#### Decoder

a third sub-layer performs multi-head attention over the output of the encoder stacks.

### Attention

- map a query and a set of key-values pairs to an output

#### Scaled dot-product attention

<img src="pictures\image-20250427110019113.png" alt="image-20250427110019113" style="zoom:67%;" />

<img src="pictures\image-20250427173806174.png" alt="image-20250427173806174" style="zoom:67%;" />

- additive attention
- ( dot-product ) multiplicative  attention：使用缩放点积注意力既能够使用到快速矩阵乘法的好处又能够避免由于当d_k太大的时候softmax造成的梯度消失的问题。
- 当d_k较小的时候，两种方法效果差不多，而当d_k较大的时候，加性注意力表现的比乘性注意力更好。

<img src="pictures\image-20250427181234204.png" alt="image-20250427181234204" style="zoom: 80%;" />

<img src="pictures\image-20250427181434900.png" alt="image-20250427181434900" style="zoom: 67%;" />

#### Multi-head attention

<img src="pictures\image-20250427175053701.png" alt="image-20250427175053701" style="zoom: 67%;" />

<img src="pictures\image-20250427181758550.png" alt="image-20250427181758550" style="zoom:67%;" />

<img src="pictures\image-20250427181831701.png" alt="image-20250427181831701" style="zoom: 67%;" />

#### **Applications of Attention in our Model**

- encoder-decoder attention
- encoder self attention
- decoder self attention

#### **Position-wise Feed-Forward Networks**

<img src="pictures\image-20250428135610242.png" alt="image-20250428135610242" style="zoom: 80%;" />

- 跨位置参数相同但是不同层使用独立权重
- 给每个token引入非线性表示，增加表示能力
- 逐个位置进行，与注意力可并行，保持整体并行友好

#### **Embeddings and Softmax**

- 两个嵌入层和softmax前线性层之间我们使用的是同样的权重，并且都乘以{d_model}^{1/2}来保持数值尺度

#### **Positional Encoding**

<img src="pictures\image-20250428141259544.png" alt="image-20250428141259544" style="zoom:67%;" />
$$
因为对于任何固定的偏移量k,PE_{pos+k}可以表示为PE_{pos}的线性函数。
$$

$$
1. 写出单一频率下的编码 \\
设
\omega_{i} = 10000^{-2i/d_{\text{model}}} \quad \text{(固定常数)} \\
则第 i 对正余弦为 \\

\begin{cases}
\text{sin分量:} & s_{i}(pos) = \sin(pos \omega_{i}) \\
\text{cos分量:} & c_{i}(pos) = \cos(pos \omega_{i}) \\
\end{cases} \\

2. 对位置偏移 k 应用三角恒等式 \\

\begin{aligned}
s_{i}(pos + k) &= \sin((pos + k)\omega_{i}) \\
&= \sin(pos \omega_{i}) \cos(k \omega_{i}) + \cos(pos \omega_{i}) \sin(k \omega_{i}) \\
c_{i}(pos + k) &= \cos((pos + k)\omega_{i}) \\
&= \cos(pos \omega_{i}) \cos(k \omega_{i}) - \sin(pos \omega_{i}) \sin(k \omega_{i}) \\
\end{aligned} \\

3.写成矩阵乘法——典型的线性变换 \\


\begin{bmatrix}s_i(pos+k)\\c_i(pos+k)\end{bmatrix}=\underbrace{\begin{bmatrix}\cos(k\omega_i)&\sin(k\omega_i)\\-\sin(k\omega_i)&\cos(k\omega_i)\end{bmatrix}}_{M_i(k)}\begin{bmatrix}s_i(pos)\\c_i(pos)\end{bmatrix} \\

·M_i(k)是 2×2 旋转矩阵，只依赖于偏移量k和 频率\omega_i,与具体的pos无关。\\

·对所有维度把这类M_i(k)组成 块对角矩阵M(k),就得到整条位置向量的关系：
PE_{pos+k}=M(k)\:PE_{pos}.
$$

<img src="pictures\image-20250501112728575.png" alt="image-20250501112728575" style="zoom:50%;" />

- 输出结果就是value的加权和，权重为query与key之间的相似度

- 三个attention

<img src="pictures\image-20250501112922984.png" alt="image-20250501112922984" style="zoom:50%;" />

- 前馈神经网络中，各个`MLP`的权重是一致的，可以point wise的去对每个输出进行操作，因为attention机制已经采集了序列中全部的信息。但是在RNN中，得一步一步的传递序列信息，不像attention是一次就采集全部的信息，但是attention里面没有包含位置信息，所以需要加入位置编码。

<img src="pictures\image-20250501135256944.png" alt="image-20250501135256944" style="zoom:50%;" />



![image-20250501190819941](pictures\image-20250501190819941.png)

# Batch normalization and Layer normalization

<img src="pictures\image-20250501190032765.png" alt="image-20250501190032765" style="zoom:50%;" />

<img src="pictures\image-20250430132651994.png" alt="image-20250430132651994" style="zoom:33%;" />

- 归一化作用：特征输入到激活函数之前，避免数据波动太大进而进入饱和区造成梯度消失
- 归一化的形式  可以使用a和b来控制归一化后满足的特定分布
- Layer normalization适用于**变长**数据
- batch normalization换一批batch后，数据针对同一特征的均值和方差可能会很大；而且如果新的句子很长在训练过程中没有遇到过原来的均值和方差就不再适用
- Layer normalization不需要记录全局的均值和方差计算更加方便，而且只针对于单个样本，更加稳定

- 对于Laye normalization切片后可以指定归一化的shape，可以选择的是对最后一个维度归一化还是对正片数据进行归一化

<img src="C:\Users\LX\Desktop\学习\LLM-paper\pictures\image-20250501101057975.png" alt="image-20250501101057975" style="zoom: 50%;" />

<img src="pictures\image-20250501101143483.png" alt="image-20250501101143483" style="zoom:50%;" />



# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- 通过超参数计算科学系的参数的量

<img src="pictures\image-20250501153423594.png" alt="image-20250501153423594" style="zoom:67%;" />

- 使用WordPiece，**WordPiece** 将罕见词拆成高频子词

- **MLM**：带掩码的大语言模型
- **NSP**：下一个句子的预测任务
- features based：ELMo
- fine-tuning：GPT，但是使用的是从左到右的框架

<img src="pictures\image-20250501154038948.png" alt="image-20250501154038948" style="zoom:67%;" />

<img src="pictures\image-20250501154317195.png" alt="image-20250501154317195" style="zoom:67%;" />

- 预训练：使用不带标记的数据进行训练
- 微调：针对特定的下游任务，使用预训练完的模型再对特定的带标记的数据进行训练



## questions

1. **BERT分为哪两种任务，各自的作用是什么；** 

- Masked Language Model（MLM，遮盖语言模型）：这种任务的目的是预测句子中部分单词的原始形式。在训练过程中，BERT模型会随机选择一些单词并用“【MASK】”标记替换它们。模型的任务是预测被替换的单词的原始形式。**这种方法可以使模型在理解句子语义的同时学习到词语之间的关系。** 

- Next Sentence Prediction（NSP，下一句预测）：这种任务的目的是预测一个句子是否是另一个句子的下一句。在训练过程中，BERT模型会从两个句子中选择一个随机的句子对，并根据是否是下一句来训练模型。**这种方法可以使模型更好地理解上下文之间的关系。**

2. **在计算MLM预训练任务的损失函数的时候，参与计算的Tokens有哪些？是全部的15%的词汇还是15%词汇中真正被Mask的那些tokens？** 

在 MLM 预训练中，只有那 **15 % 被抽中用来预测的 token 位置** 参与交叉熵损失的计算——无论它们在输入里被替换成 `[MASK]`、随机词，还是保持不变；其他 85 % 位置被设为 `-100/ignore_index`，完全不计入 loss。

<img src="pictures\image-20250501172610691.png" alt="image-20250501172610691" style="zoom: 67%;" />

3. **在实现损失函数的时候，怎么确保没有被 Mask 的函数不参与到损失计算中去；** 

把没抽中的 85 % 标签写成 `-100` 并把 `ignore_index` 传给交叉熵，就能保证这些 token 对损失与梯度“完全隐形”。 

<img src="pictures\image-20250501173301884.png" alt="image-20250501173301884" style="zoom:67%;" />

4. **BERT的三个Embedding为什么直接相加** 

把三个 embedding 直接**相加**是一种在 **维度兼容性、计算效率、信息分离性** 之间折中的工程选择：既不会膨胀模型规模，又足以让后续层学到如何利用语义、句段与位置信息。

高维随机向量几乎天然正交；把三块信息初始化在同一空间后相加，信息仍可在后续线性变换中被分离，**几乎不丢失可分性**。

<img src="pictures\image-20250501174131063.png" alt="image-20250501174131063" style="zoom:67%;" />

5. **BERT的优缺点分别是什么？** 

**优点：**

- **深度双向语义建模**：通过 **Masked Language Modeling (MLM)**，每个词都能同时利用左右上下文，捕捉更丰富的语义线索，而不像传统 LM 只能单向扫描。
- **迁移学习效率高**：预训练后只需在输出端加一层任务头即可微调，多数任务几分钟到几小时就能收敛，对小数据集尤其友好。
- **子词 WordPiece 解决 OOV**：WordPiece 将罕见词拆成高频子词，显著缓解稀疏与 OOV 问题，对黏着语和新词更鲁棒。
- **SOTA 与广泛验证**：在 GLUE、MultiNLI、SQuAD 等公开基准长期保持顶级成绩，成为后续模型（RoBERTa、ALBERT、DeBERTa）的基线。

**缺点：**

- **训练与推理成本高**
- **输入长度受限（512 tokens）:**预训练时位置 Embedding 只到 512，注意力矩阵 O(n²) 也让显存在长序列上爆炸；若要处理报告/论文等长文需截断、滑窗或改用 Longformer、BigBird 等长上下文模型。
- **MLM 与下游分布不一：**:预训练阶段加入了特殊的 `[MASK]` 标记，下游推理却不会出现，存在 **pretrain–finetune gap**；Google 论文中用 10 % 保留原词、10 % 替换随机词来弱化此差距，但仍不是根治方案。

6. **你知道有哪些针对BERT的缺点做优化的模型？** 

- 轻量化与蒸馏：算力 / 延迟优化

| 模型           | 关键手段                            | 效果概览                                       |
| -------------- | ----------------------------------- | ---------------------------------------------- |
| **DistilBERT** | 全过程知识蒸馏                      | 参数-40 %，推理快 60 %，GLUE 损失≈3 %oaicite:0 |
| **ALBERT**     | 词表分解 + 层间参数共享             | BERT-large 同级准确率，参数↓90 %oaicite:1      |
| **TinyBERT**   | 两阶段蒸馏（预训练 + 下游）         | 模型 7× 更小、9× 更快，精度保持 96 %oaicite:2  |
| **MobileBERT** | 瓶颈-Inverted Bottleneck + 逐层蒸馏 | 4.3× 更小、5.5× 更快，Pixel 手机延迟 62 ms:    |

- 改进预训练目标：降低 [MASK] 差距 / 提高效率

| 模型        | 新目标                           | 针对问题                                                     |
| ----------- | -------------------------------- | ------------------------------------------------------------ |
| **RoBERTa** | 去掉 NSP、动态掩码、大批量大数据 | BERT “欠训练” 与 NSP 噪声oaicite:4                           |
| **ELECTRA** | Replaced Token Detection（RTD）  | MLM 只训练 15 % 位置 → 采样效率低；RTD 全 token 监督更快更准oaicite:5 |

- 稀疏 / 结构化注意力：突破 512-token 限制

| 模型           | 注意力模式             | 复杂度                                                       |
| -------------- | ---------------------- | ------------------------------------------------------------ |
| **Longformer** | 滑窗 + 全局 token      | O(n · w) 近线性，支持万字文档oaicite:6                       |
| **BigBird**    | 块-稀疏 + 随机 + 全局  | 理论上具备 Transformer 近似表达能力；O(n) 内存/算力oaicite:7 |
| **Reformer**   | LSH-attention + 可逆层 | O(n log n) 注意力 + 常数显存回传oaicite:8                    |

- 架构革新：内容-位置解耦与表达能力

| 模型        | 核心改动                                                     | 收益                                                         |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **DeBERTa** | Disentangled Attention（独立建模内容 & 位置）+ Enhanced Mask Decoder | 在半数语料下超 RoBERTa-Large，并首个单模型超越 SuperGLUE 人类基线oaicite:9 |

- 鲁棒性与公平性：对抗训练与去偏

| 方法 / 模型             | 解决痛点             | 代表结果                                                     |
| ----------------------- | -------------------- | ------------------------------------------------------------ |
| **FreeLB**              | 多步对抗扰动提升泛化 | 在 GLUE、ARC 均刷新 SOTA，同时显著增强抗对抗样本性能oaicite:10 |
| **Debiasing-BERT 系列** | 性别 / 种族刻板映射  | 多层约束（数据重采样 + 对抗损失）显著降低偏见指标oaicite:11  |

7. **BERT怎么用在生成模型中？**

**把 BERT 用到生成任务的核心在于：**要么修改掩码和预训练目标，**让同一套参数既能双向理解又能单向/Seq2Seq 解码**；要么利用 BERT 强判别能力，在并行填空、扩散去噪或可控生成中担当“迭代修正器”或“输出审计员”。随着 UniLM/BART/GLM 等模型与非自回归策略的成熟，BERT 家族已能覆盖绝大多数文本生成应用，并在并行效率、长文本处理和可控性上提供与 GPT 系不一样的工程优势。

**1 . 为什么 BERT 输入端要包含 token、segment、position 三个 embedding？三者如何相加？**
 BERT 需要：

- **Token Embedding** 表示词/子词本身；
- **Segment Embedding**（A/B=0/1）告诉模型同一句对里的句子编号，用来完成 NSP 任务；
- **Position Embedding** 让 Transformer 能区分顺序信息。
   这三种向量都被投到同一维度 *d*（768/base 或 1024/large），然后做逐元素相加后再送入第一层 Transformer；这样就能保持残差、Layer Norm 等结构不变，同时把 3 种信息融合进同一向量空间。 ([BERT - The purpose of summing token embedding, positional ...](https://datascience.stackexchange.com/questions/108740/bert-the-purpose-of-summing-token-embedding-positional-embedding-and-segment?utm_source=chatgpt.com), [BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

------

**2 . 如果把三种 embedding 改为拼接再投影，会带来哪些代价？**
 拼接后维度变为 3 × *d*，第一层必须增加一个线性投影把 3 *d* 压回 *d*，导致参数量和 FLOPs ≈ 3 倍，显存与延迟同步上涨；同时所有残差路径和权重共享假设都会被破坏，收敛更难。 ([BERT - The purpose of summing token embedding, positional ...](https://datascience.stackexchange.com/questions/108740/bert-the-purpose-of-summing-token-embedding-positional-embedding-and-segment?utm_source=chatgpt.com), [BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

------

**3 . WordPiece 词表是如何构造的，为什么能缓解 OOV？**
 WordPiece 自底向上统计最常共现的字串对并迭代合并，直到达到预设词表大小；长低频词会被拆成高频子词，使任何新词都能拆成已知片段，从而显著降低 OOV 与稀疏问题。 ([WordPiece: Subword-based tokenization algorithm | Chetna - Medium](https://medium.com/towards-data-science/wordpiece-subword-based-tokenization-algorithm-1fbd14394ed7?utm_source=chatgpt.com), [BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

------

**4 . Base=12 层 768 维 / Large=24 层 1024 维是如何选择的？把隐藏维扩到 4096 会遇到什么瓶颈？**
 作者在 BERT 论文附录用网格实验验证到 768/12 与 1024/24 在 GLUE 和 SQuAD 上收益与成本平衡最佳；*d* 和层数再增大时，参数量与计算复杂度以 *L·d²* 线性/平方上升，显存爆炸、训练时间线性拉长且更易梯度爆炸/过拟合。 ([BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

------

**5 . 解释 15 % 抽样 + 80-10-10 替换策略的设计初衷。**
 先对序列做伯努利抽样，15 % 位置被标为预测目标；其中 80 % 换 `[MASK]` 让模型学习填空，10 % 换随机词增加噪声防止只依赖 `[MASK]`，10 % 保持原词使预训练-推断分布更接近。 ([BERT MLM - 80% [MASK\], 10% random words and 10% same word](https://stats.stackexchange.com/questions/575002/bert-mlm-80-mask-10-random-words-and-10-same-word-how-does-this-work?utm_source=chatgpt.com), [BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

------

**6 . 在实现损失时，如何保证只有那 15 % 位置产生梯度？**
 把 `labels` 复制自输入，再把 `~masked_indices` 位置写成 `-100`（PyTorch 默认 `ignore_index`）；`CrossEntropyLoss(ignore_index=-100)` 会忽略这些位置，因此未被抽中的 85 % 不参与反向传播。 ([Intricacies of nn.CrossEntropyLoss Ignore Index and Gradients](https://jkschin.com/blog/2023/cross-entropy-loss-ignore-index/?utm_source=chatgpt.com), [Use of ignore_index on CrossEntropyLoss() for text models](https://stats.stackexchange.com/questions/502750/use-of-ignore-index-on-crossentropyloss-for-text-models?utm_source=chatgpt.com))

------

**7 . NSP 任务流程是什么？RoBERTa 为什么去掉 NSP？**
 NSP 让 `[CLS]` 表征对句子对 (A,B) 判断 “B 是否紧跟 A”；正样本来自语料原句对，负样本随机拼接。RoBERTa 通过消融发现 NSP 信号噪声大且限制批大小，去掉后总体效果反而更好。 ([Next sentence prediction in RoBERTa - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/76872/next-sentence-prediction-in-roberta?utm_source=chatgpt.com), [RoBERTa: A Robustly Optimized BERT Pretraining Approach - arXiv](https://arxiv.org/abs/1907.11692?utm_source=chatgpt.com))

------

**8 . Whole Word Masking (WWM) 的思想及改进点**
 WWM 在中文等拼写无空格语言中一次遮盖完整词，而不是逐字子词，从而让 MLM 更符合语义单元、减少信息泄漏并带来 1-2 pt F1 提升。 ([Pre-Training with Whole Word Masking for Chinese BERT - arXiv](https://arxiv.org/abs/1906.08101?utm_source=chatgpt.com))

------

**9 . 为什么原文选择 AdamW+Weight Decay 而不是传统 Adam+L2？**
 Adam 的动量会把 L2 正则当作梯度加权，导致实际权重衰减不稳定；AdamW 将 weight-decay 与梯度更新解耦，可在大 batch 下稳定训练并减少过拟合。 ([Understanding L2 regularization, Weight decay and AdamW](https://benihime91.github.io/blog/machinelearning/deeplearning/python3.x/tensorflow2.x/2020/10/08/adamW.html?utm_source=chatgpt.com))

------

**10 . Layer-wise Learning-Rate Decay (LLRD) 微调有什么好处？**
 LLRD 为靠近输出层的参数设较大学习率，底层设较小学习率，可在保留通用语义的同时快速适配任务，显著减少灾难性遗忘。 ([[PDF\] arXiv:2212.06138v1 [cs.CV] 12 Dec 2022](https://arxiv.org/pdf/2212.06138?utm_source=chatgpt.com))

------

**11 . Dropout 与 Attention Dropout 在 BERT 中的默认值及调参经验**
 原始实现均设为 0.1；实战中可在小数据集将 Dropout 提到 0.2-0.3 抑制过拟合，在大规模任务或蒸馏模型中降到 0.05 获取更快收敛。 ([BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

------

**12 . DistilBERT、ALBERT、TinyBERT 的核心压缩策略对比**

- **DistilBERT**：跳层蒸馏，层数减半（12→6），参数-40 %，推理快 60 %。 ([Papers Explained 06: Distil BERT - Medium](https://medium.com/dair-ai/papers-explained-06-distil-bert-6f138849f871?utm_source=chatgpt.com))
- **ALBERT**：词表分解 + 各层参数共享，参数-90 %，精度接近 BERT-Large。 ([ALBERT: A Lite BERT for Self-supervised Learning of Language ...](https://arxiv.org/abs/1909.11942?utm_source=chatgpt.com))
- **TinyBERT**：两阶段蒸馏（预训练+下游），尺寸仅 1/7，仍保留 96 % GLUE 分数。 ([TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351?utm_source=chatgpt.com))

------

**13 . Self-Attention 的时空复杂度为何是 O(n²)？如何降低？**
 全序列 Q × Kᵀ 生成 n × n 相似度矩阵，两次与 V 乘法共 O(n²·d)；稀疏滑窗（Longformer）或块+随机+全局稀疏（BigBird）将每个 token 只与 O(1) 或 O(w) 邻居交互，使复杂度降到近线性。 ([Computational Complexity of Self-Attention in the Transformer Model](https://stackoverflow.com/questions/65703260/computational-complexity-of-self-attention-in-the-transformer-model?utm_source=chatgpt.com), [[2004.05150\] Longformer: The Long-Document Transformer - arXiv](https://arxiv.org/abs/2004.05150?utm_source=chatgpt.com), [[2007.14062\] Big Bird: Transformers for Longer Sequences - arXiv](https://arxiv.org/abs/2007.14062?utm_source=chatgpt.com))

------

**14 . 把 BERT 部署到移动端时，8-bit 量化与蒸馏各有什么利弊？**
 量化无需重新训练即可减半显存并借助 INT8 运算，精度损失通常 < 1 pt，但硬件需支持向量 INT8；蒸馏能同时减参数并保持浮点精度，但需额外教师-学生训练周期。 ([Papers Explained 06: Distil BERT - Medium](https://medium.com/dair-ai/papers-explained-06-distil-bert-6f138849f871?utm_source=chatgpt.com), [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351?utm_source=chatgpt.com))

------

**15 . BERT 的长度上限为何是 512？如何突破？**
 位置嵌入只预训练到 512，且全量注意力显存 O(n²)；Longformer 通过窗口注意力，BigBird 通过块稀疏+随机注意力，把记忆与计算降到 O(n) 并支持上万 token。 ([[2004.05150\] Longformer: The Long-Document Transformer - arXiv](https://arxiv.org/abs/2004.05150?utm_source=chatgpt.com), [[2007.14062\] Big Bird: Transformers for Longer Sequences - arXiv](https://arxiv.org/abs/2007.14062?utm_source=chatgpt.com))

------

**16 . BERT 在多句长对话中语义漂移的原因？**
 Segment 仅有 0/1，无法标记第三句起的轮次；缺乏显式对话状态导致模型难以追踪多轮 referent，从而出现语义漂移。 ([BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

------

**17 . RoBERTa 对 “BERT 欠训练” 做了哪三项修订？**

1. 去掉 NSP；2. 动态掩码；3. 更大 batch、更长训练、更大语料（160 GB）。这些改动在 GLUE、SQuAD 提升 1-2 pt。 ([RoBERTa: A Robustly Optimized BERT Pretraining Approach - arXiv](https://arxiv.org/abs/1907.11692?utm_source=chatgpt.com), [[PDF\] arXiv:1907.11692v1 [cs.CL] 26 Jul 2019](https://arxiv.org/pdf/1907.11692?utm_source=chatgpt.com))

------

**18 . ELECTRA 的 Replaced Token Detection 如何提升采样效率？**
 ELECTRA 用小型生成器替换部分 token，再让判别器区分 “真/假”；判别器每个位置都有监督，训练速度与样本效率比 MLM 高 4× 以上，同算力下效果优于 BERT。 ([More Efficient NLP Model Pre-training with ELECTRA](https://research.google/blog/more-efficient-nlp-model-pre-training-with-electra/?utm_source=chatgpt.com))

------

**19 . 把 BERT 改造成生成模型的三条技术路线**

- **UniLM 系列**：通过可切换 self-attention mask，让同一模型既能双向编码又能单向/Seq2Seq 解码。 ([BERT - The purpose of summing token embedding, positional ...](https://datascience.stackexchange.com/questions/108740/bert-the-purpose-of-summing-token-embedding-positional-embedding-and-segment?utm_source=chatgpt.com), [UniLMv2: Pseudo-Masked Language Models for Unified ... - arXiv](https://arxiv.org/abs/2002.12804?utm_source=chatgpt.com))
- **BART/MASS/GLM**：添加 Span-infill 等去噪重建目标的 Encoder-Decoder。 ([Papers Explained 09: BART - Medium](https://medium.com/dair-ai/papers-explained-09-bart-7f56138175bd?utm_source=chatgpt.com), [Next sentence prediction in RoBERTa - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/76872/next-sentence-prediction-in-roberta?utm_source=chatgpt.com))
- **非自回归迭代填空 (Mask-Predict/BPDec)**：用 BERT 并行生成全句，并多轮重写低置信 token。 ([WordPiece: Subword-based tokenization algorithm | Chetna - Medium](https://medium.com/towards-data-science/wordpiece-subword-based-tokenization-algorithm-1fbd14394ed7?utm_source=chatgpt.com))

------

**20 . 举例说明 BERT 的性别偏见，并列出两种缓解策略**

- 关联测试显示 “医生—男性” 概率远高于 “医生—女性”，反映训练语料刻板印象。 ([Measuring and Mitigating BERT's Gender Bias - ACL Anthology](https://aclanthology.org/2020.gebnlp-1.1/?utm_source=chatgpt.com))
- 缓解可用 (a) 数据再采样/反向标注平衡正负；(b) 对抗训练或下降梯度消除性别信息于中间层。 ([Measuring and Mitigating BERT's Gender Bias - ACL Anthology](https://aclanthology.org/2020.gebnlp-1.1/?utm_source=chatgpt.com), [[PDF\] FREELB: ENHANCED ADVERSARIAL TRAINING FOR NATURAL ...](https://openreview.net/pdf?id=BygzbyHFvB&utm_source=chatgpt.com))

------

**21 . 推导 Fused QKV 线性层可节省多少参数，并给出 BERT-Large 规模**
 把三个独立 `W_Q,W_K,W_V ∈ ℝ^{d×d}` 融合成单个 `W_QKV ∈ ℝ^{d×3d}` 仅在实现上减少三次 GEMM→一次 GEMM，参数量本质不变仍为 3 d²≈ 3·1024² ≈ 3.1 M，但可省 I/O 与缓存开销、提升吞吐。 ([Computational Complexity of Self-Attention in the Transformer Model](https://stackoverflow.com/questions/65703260/computational-complexity-of-self-attention-in-the-transformer-model?utm_source=chatgpt.com))

------

**22 . 用数学证明“高维随机向量近似正交”意味着三 embedding 相加信息损失可忽略**
 根据 Johnson-Lindenstrauss 引理，*k* 维随机投影几乎保持点间距离；当 *d*≫1 时，任意两随机向量夹角趋近 90°，三 embedding 在求和后仍可由线性层再分离，信息损失可忽略。 ([BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com), [BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

------

**23 . Gradient Checkpointing 在 BERT 训练中的存储-算力权衡**
 通过只存储部分层的激活，反向时再计算一次前向，显存占用可降 30-50 %，代价是推理 FLOPs ≈ 1.5-2×；适用于大模型或长序列场景。 ([BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

------

**24 . 如果把 `[CLS]` 表征改为平均池化，哪些下游任务受益 / 受损？**
 序列级任务（情感分类、句子对匹配）中 `[CLS]` 经训练学习到专属位置信号，通常优于简单均值；而序列标注或检索场景中平均池化能更好反映全局语义，避免过度依赖首位 token。 ([BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

------

**25 . mBERT 在非平衡语料上的零样本跨语种迁移现象**
 mBERT 共享词片与参数空间，低资源语言能通过形似词干获得间接监督；但高资源语言主导梯度导致语义空洞，表现为欧语间迁移准确率高，至隔系语族显著下降。解决方案包括对低资源语重采样或使用语言适配层。 ([BERT: Pre-training of Deep Bidirectional Transformers for Language ...](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com))

# **Mission: Impossible Language Models**

## :question: 研究问题 

近期一些语言学家声称，LLM 在“可能”和“根本不可能”的语言上没有差别，因此无法为语言学提供线索。但几乎没有实证数据支撑；作者要问的是：**LLM 真的能像人一样区分“可能／不可能语言”吗？**

## :point_up_2: 方法

**构造“不可能语言”光谱**

- 基于英语 BabyLM 数据，设计三大类变体：
   *Shuffle*（随机/局部/确定性打乱词序）、*Reverse*（全句或部分反转）、*Hop*（动词变形靠“数第 n 个词”标记）。​​
- 每类配一个“NO*”控制语言，几乎等同英语。

<img src="pictures\image-20250504190643695.png" alt="image-20250504190643695" style="zoom:50%;" />

**训练 & 评测**

- 从零开始训练多组 GPT-2-small（含/不含位置编码），每种语言 5 个随机种子。
- **Experiment 1**：看测试集困惑度（perplexity）随训练步数变化。
- **Experiment 2**：对 *Hop* 类做两项 surprisal 测试，检查模型是否“期待”正确的计数标记。
- **Experiment 3**：用因果抽象分析（interchange intervention）探内部机制。

## :pager: 结果

总体而言，GPT-2 **明显更快、更好地学会“可能语言”**，对线性计数、全局打乱等违反层级/局域性的规则学习效率低，动摇了“LLM 全能”论断。

## :qatar: 细节

<img src="pictures\image-20250504165707363.png" alt="image-20250504165707363" style="zoom:50%;" />

:factory: 一种“不可能语言”

- **只看线性顺序**——例如“凡是排在第 n 个位置的词必须带某个后缀”。
- **只看线性顺序**——例如“凡是排在第 n 个位置的词必须带某个后缀”。

:red_circle: 这两个 -s 的出现完全忽视了树形结构，只依赖“它在整个句子里的第几词”——这正是“不可能语言”式的规则。

- **自然语言** 的语法/形态机制，总是依赖 **层级结构 (hierarchical structure)**，而不是简单的线性计数。
- 如果某种“语言”要求使用线性计数或固定词位来决定形态变化，那就违反了人类语言的普遍约束，因此被称为“不可能语言”。
- 通过这样的对比，研究者可以在实验中检验大语言模型（或人类）能否区分“可能”和“明显不可能”的语法机制。

------

<img src="pictures\image-20250504183445216.png" alt="image-20250504183445216" style="zoom:50%;" />

:question: **GPT-2 这类语言模型在学习“可能—不可能”语言的难度梯度上，能否表现出和人类相似的偏好？**

<img src="pictures\image-20250504191700358.png" alt="image-20250504191700358" style="zoom: 67%;" />

<img src="pictures\image-20250504191855692.png" alt="image-20250504191855692" style="zoom: 67%;" />

------

<img src="pictures\image-20250504184300885.png" alt="image-20250504184300885" style="zoom:50%;" />

:question: shuffle语言构造规则

**impossibility continuum 不可能性连续谱**

<img src="pictures\image-20250504192425123.png" alt="image-20250504192425123" style="zoom: 67%;" />

------

<img src="pictures\image-20250504184833557.png" alt="image-20250504184833557" style="zoom:50%;" />

论文把英语改造成三种 *HOP* 语言，用特殊标记 **S/P** 代替三单 **-s**：

- **NOHOP** 标记紧跟在动词后（最像英语，控制组）

- **TOKENHOP** 标记放在动词后 **4 个 token** 处

- **WORDHOP** 标记放在动词后 **4 个单词** 处
   后两种必须“数”token/单词，属典型“线性计数”规则，被视为“不可能语言”的温和版本​​。

<img src="pictures\image-20250504193538853.png" alt="image-20250504193538853" style="zoom: 67%;" />

<img src="pictures\image-20250504194318662.png" alt="image-20250504194318662" style="zoom: 67%;" />

------

<img src="pictures\image-20250504184926256.png" alt="image-20250504184926256" style="zoom:50%;" />

:point_up_2: 给模型两条输入（**base** 与 **source**），把某一层、某一位置的隐藏向量从 source 复制到 base，再继续前向传播，看输出会不会随之改变。如果“换血”后 base 的输出像 source，那就说明 **被替换的隐藏向量携带并因果地控制了目标信息**。

幻灯片右边的示意图

- **橙色**：base 句 *The man be …*（动词应加 P，因为 *man* 单数？或因 Hop 规则？）
- **绿色**：source 句 *The men be …*（或已满足计数条件）
- 紫色虚线：把第 *L* 层、第 *t* 个 token 的隐藏状态从绿色网络插到橙色网络。
- **若注入后，橙色网络最终在动词处成功输出标记 P，就说明那层那位存放了“该不该加 P”的关键信息。**

**Interchange-Intervention Accuracy (IIA)**
 在所有测试句上做一次全网格搜索（所有层 × 所有 token 位置），统计成功率作为 IIA 热图。数值越高→信息越集中、越可预测。

<img src="pictures\image-20250504194513128.png" alt="image-20250504194513128" style="zoom:67%;" />

------

<img src="pictures\image-20250504185031825.png" alt="image-20250504185031825" style="zoom:50%;" />

:3rd_place_medal: **语言与模型都偏爱“信息局域性”**

<img src="pictures\image-20250504194719368.png" alt="image-20250504194719368" style="zoom:67%;" />

GPT-2 之所以把“可能语言”当成“好学的语言”，核心在于双方共享**“信息局部化”**这一深层偏好；而反局域、需线性计数的规则既罕见于人类语言，也让模型学得费劲。
