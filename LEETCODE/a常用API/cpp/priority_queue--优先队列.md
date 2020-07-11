####  基础操作

- [ ] top  : 访问头元素
- [ ] empty
- [ ] size
- [ ] push
- [ ] emplace ： 原地构造一个元素并插入队列
- [ ] pop
- [ ] swap：交换内容

#### 定义

```c++
priority_queue<Type, Container, Functional> 
// Type : 数据类型
// Container : 容器类型，数组实现的容器，vector,deque， 默认是vector，不能是list, 
// Functional 是比较方式

priority_queue <int, vector<int>, greater<int> >  q;  // 升序, 小顶堆

priority_queue <int, vector<int>, less<int> >  q;  // 降序，大顶堆， 默认是这个



```

