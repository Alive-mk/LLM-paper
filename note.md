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



