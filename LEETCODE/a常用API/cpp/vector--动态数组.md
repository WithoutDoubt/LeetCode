#### 构造

```c++
C c;

C c1 (c2);

C c(begin, end);       

C c{a, b, c ... }; 
```

​         

#### 获取迭代器

- [ ] `c.begin( ), c.end( )`
- [ ] `c.rbegin( ), c.rend( )`                      // 不支持forward_list

```c++
c.cbegin( ), c.cend( )

// 不支持forward_list
reverse_iterator
const_reverse_iterator
c.crbegin(), c.crend()
```

 

#### 赋值

```c++
c1 = c2
c1 = {a , b, c ...}
a.swap(b)
swap(a,b)
```

 

#### 大小

```c++
c.size()
c.max_size()
c.empty()    
```

