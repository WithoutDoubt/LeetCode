#### LRU （Least Recently Used）

需求分析：

>最大容量为capacity                                        
>
>put 存入键值对  (key,val)                                        【复杂度为1】
>
>get 获取 (key, val)， 如果key 不存在则返回 -1     【复杂度为1】
>
>如果满了，删除最后一个
>
>每次访问要把数据插入到队头
>
>
>
>要求：查找快，删除快，有顺序之分
>
>哈希查找快，数据没有固定顺序
>
>插入删除快，但是查找慢。
>
>
>
>所以使用：哈希链表【哈希以及双向链表】

问题：

> 1. 为什么使用的是 双向链表
>
>    因为在快速删除节点的过程中，我们需要操作 前一个节点 prev
>
> 2. 哈希表中已经存在key, 为什么链表中还要存 键值对呢？
>



节点

```c++
struct Node{
    int key, val;
    Node* next;
    Node* prev;
}
```



```c++
unordered_map<int,Node*>map;
DoubleList* cache;

int get(int key){
    if (key 不存在){
    	return -1;
    }else{
        将数据（key,val）提到开头；
        return val；
    }
}

void put(int key, int val){
    Node* x = new Node(key, val);
    if (key 已经存在){
        旧数据删除；
        将新节点x插到开头；    
    }else{
        if (cache 已满){
            删除链表最后一个数据；
            删除map中的键；
        }
        将新节点x插到开头；
        map中新建key对新节点x的映射；    
    }
}
```

