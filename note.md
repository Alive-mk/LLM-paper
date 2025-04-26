# **Attention Is All You Need**

## abstract

```json
sequence transduction model,recurrent,convolutional,dispense,solely,be superior in,parallelizable,literature
state of the art,
```

- `BLEU`
  `BLEU`（Bilingual Evaluation Understudy）是一种用于评估机器翻译质量的指标，通过比较机器翻译输出和一组参考翻译之间的n-gram重叠来评分。

## introduction

```
numerous,preclude,inherently,sequential nature,
```

- long short-term memory neural networks
- gated recurrent neural networks
- sequence modeling and transduction
- `RNN`中其中 h_t 由上一个隐藏状态 h_{t-1} 和位置 t 的输入共同决定
- factorization tricks
- conditional computation

## background

```json
resolution,arbitary,counteract,
```

- transduction model
  `seq2seq` model
- 在传统模型中，把任意两个位置之间的信号关联起来所需要的运算量依赖着两者之间的距离，而transformer将这个计算削减至常量
- end-end memory networks

## model architecture

```json
point-wise,identical,residual,implemented,facilitate,attend to(关注),
```

- 编码器把一个输入序列映射成连续表示序列Z，然后解码器会根据Z按照一步产生一个元素的方式产生一个输出序列。解码器采用的是自回归的方式，在生成下一个符号的的时候会把前面已经生成的符号作为额外的输入。

- 堆叠的自注意力机制和逐点的全连接层。

<img src="pictures\image-20250426224402795.png" alt="image-20250426224402795" style="zoom:67%;" />

- masking
- shifted right

### encoder and decoder stacks

