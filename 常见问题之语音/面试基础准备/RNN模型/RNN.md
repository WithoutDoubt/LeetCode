Recurrent neural Network 

RNN在sequence modeling tasks 上表现很好的原因是RNN能够利用之前的信息对当前内容进行预测和判断。



RNN maintain a vector of hidden activations that are propagated through time

The intuitive appeal of recurrent modeling is that the hidden state can act as a representation of everything that has been seen so far in the sequence.

维护隐藏激活的向量，这些激活是通过时间传播的

循环建模的直观吸引力是，隐藏状态可以作为序列中到目前为止看到的所有内容的表示。



RNN的变体LSTM的出现缓解了

---

---

sequence model task 的一个关键能力是每个时刻输入之间的关系进行建模，但是香草DNN 和 CNN不能做到持续记忆，所以在进行序列建模时，很难将某一时刻的输入和其周围时刻输入的关系都考虑到，这限制了DNN 和 CNN 在进行sequence model 时的性能。

然而，RNN的出现可以缓解该问题，它是带有循环的神经网络，允许信息保留一段时间。

下图给出了一个RNN的简单结构图



由该图可以发现，RNN在预测当前时刻输出时，不仅使用了当前时刻的输入，同时也考虑到了在它之前时刻的输出。

Formally,
$$
h_t = H_h(W_h\cdot[h_{t-1},x_t]+b_h)
$$

$$
y_t = H_y(W_y\cdot h_t + b_y)
$$

$x_t$ 为输入向量，$h_t$ 是隐状态向量，$W_h$ 和 $W_y$ 是权值矩阵，$b_h$ 和 $b_y$ 是偏置向量，而$H_h$ 和 $H_y$ 是任意的激活函数。 

RNN的主要特点是隐藏状态 $h_t$ 可以作为序列中到目前为止看到的所有内容的表示。

However，当它开始学习长期依赖的时候，很容易suffer from 

Basic RNN architectures are notoriously diffificult to train and more elaborate architectures are commonly used instead, such as the LSTM (Hochreiter & Schmidhuber, 1997) and the GRU (Cho et al., 2014).

the LSTM and the GRU 可以缓解梯度消失和梯度爆炸的问题，但是仍然达不到长期依赖。

同时，RNN及其变体网络因为固有的结构设计：前一个时刻的输出将用作最后一个时刻的输入，因此RNNs不能并行化顺序计算过程，计算速度慢

the output of the previous moment in the recurrent network will be used as the input of the last moment, so it cannot parallelize the sequential calculation process, and the calculation speed is slow.



---

A key requirement of sequence modeling is to model the relationship between the input at each moment, but DNNs and CNNs cannot achieve continuous memory. 

Therefore, it is difficult to consider the input and The relationship entered around. 

This limits the performance of DNN and CNN when performing a sequence model.

However, the Recurrent Neural Networks (RNNs) can alleviate this problem. 

It is a neural network with loops that allows information to be retained for a period of time.

The model architecture of RNNs is shown in `\ref{fig1} `

【图】

It can be found from this figure that when RNN predicts the output at the current time, it not only uses the input at the current time, but also takes into account the output at the time before it.



















