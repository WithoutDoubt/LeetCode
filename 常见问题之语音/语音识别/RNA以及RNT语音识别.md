RNT 和 CTC 也是在training的时候加个 alignment，穷举所有可能的alignment

#### RNT

ht 作为input，一直当做输入，然后 直到 输出的是 空，才停止

![1594795732838](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\1594795732838.png)

[紫色] language Model 只接受 text，

It is critical for training algorithm，

Language Model： ignore speech，only consider tokens

---

