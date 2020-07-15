For online streaming speech recognition， use uni-directional RNN

#### CTC 

- 只有一个 Encoder
- Input T acoustic features，output T tokens （ignoring down sampling）
- Output tokens including $\nu$ , merging duplicate tokens, removing $\nu$ .

![1594794304653](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\1594794304653.png)

---

paired training data :               alignment 对齐

x1,x2,x3,x4 好棒                   x1,x2,x3,x4    好 $\nu$  $\nu$ 棒  

有很多种预处理的方法，实际上是使用所有可以的paired training data

