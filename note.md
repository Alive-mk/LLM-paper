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

# Batch normalization and Layer normalization

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



7. **BERT怎么用在生成模型中？**
