#### 常见模型

都是 encoder - decoder

- LAS

  > Encoder 将 语音$[x_1,x_2,...x_n]$ 中与文本有关的特征提取出来
  >
  > ​	1-D  CNN 是靠谱的 和 RNN 是靠谱的
  >
  > ​	常用的是 CNN + RNN
  >
  > ​	或者是 self-attention
  >
  > Listen--Down Sampling
  >
  > ​	Pyramid RNN
  >
  > ​	Pooling over time
  >
  >  	Dilated CNN --- Time-delay DNN (TDNN)
  >
  > ​	Truncated Self-attention  截短的自我注意
  >
  > Attention
  >
  > ​	$Z^0$ 作为输入
  >
  > Decoder
  >
  > ​	$C^0$ 作为 input， 得到一个 Distribution over all tokens
  >
  > ​	给每一个token 一个几率，决定输出哪一个token
  >
  > Beam Search is usually

- CTC

- RNN-T

- Neural Transducer

- Monotonic Chun

