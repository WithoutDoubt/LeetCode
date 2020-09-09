Java 常用的集合、函数等

首先是

- 排序
- 动态数组
- HashMap
- 字符串
- 队列
- 栈
- 优先队列
- 回溯                    最重要（只要是动态规划，就可以用回溯做）



#### 排序

---

#### 动态数组（集合中的List部分）

1. 理论上：

   - [ ] Java.util.Collection

   - [ ] Collection 接口下有三个接口：Set、List、Queue

   - [ ] Set 接口对应的实现类：HashSet、LinkedHashSet、TreeSet

   - [ ] Queue 接口对应的实现类：LinkedList, PriorityQueue

   - [ ] List 接口对应三个实现类：ArrayList、Vector、LinkedList

     - [ ] ArrayList：

       > 优点：底层数据结构是数组，查询块，增删慢
       >
       > 缺点：线程不安全，效率高​ 

     - [ ]  Vector：

       > 优点：底层数据结构是数组，查询块，增删慢
       >
       > 缺点：线程安全，效率低

     - [ ] LinkedList：

       > 优点：底层数据结构是链表，查询慢，增删快
       >
       > 缺点：线程不安全，效率高

2. 常用方法 ArrayList：

   - [ ] 构造方法：

     > `List<Integer> list = new ArrayList<>();`
     >
     > `List<Integer> list = new ArrayList<>(Collection<? extends E> c );`

   - [ ] 常用方法：增删改查

     | Modifier and Type | Method and Description                                       |
     | ----------------- | ------------------------------------------------------------ |
     | `boolean`         | `add(E e)`                                                                                                             将指定的元素追加到此列表的末尾。 |
     | `void`            | `add(int index, E element) `                                                                         在此列表中的指定位置插入指定的元素。 |
     | `boolean`         | `addAll(Collection<? extends E> c) `                                                        按指定集合的Iterator返回的顺序将指定集合中的所有元素追加到此列表的末尾。 |
     | `boolean`         | `addAll(int index, Collection<? extends E> c)`                                        将指定集合中的所有元素插入到此列表中，从指定的位置开始。 |
     | `void`            | `clear()                                                            `                                                                                                               从列表中删除所有元素。 |
     | `Object`          | `clone()`                                                                                                               返回此 `ArrayList`实例的浅拷贝。 |
     | `boolean`         | `contains(Object o) `                                                                                        如果此列表包含指定的元素，则返回 `true` 。 |
     | `void`            | `ensureCapacity(int minCapacity)`                                                                               如果需要，增加此 `ArrayList`实例的容量，以确保它可以至少保存最小容量参数指定的元素数。 |
     | `void`            | `forEach(Consumer<? super E> action)`                                                    对 `Iterable`的每个元素执行给定的操作，直到所有元素都被处理或动作引发异常。 |
     | `E`               | `get(int index)`                                                                                                返回此列表中指定位置的元素。 |
     | `int`             | `indexOf(Object o)`                                                                                          返回此列表中指定元素的第一次出现的索引，如果此列表不包含元素，则返回-1。 |
     | `boolean`         | `isEmpty()`                                                                                                          如果此列表不包含元素，则返回 `true` 。 |
     | `Iterator<E>`     | `iterator()`                                                                                                        以正确的顺序返回该列表中的元素的迭代器。 |
     | `int`             | `lastIndexOf(Object o)`                                                                                             返回此列表中指定元素的最后一次出现的索引，如果此列表不包含元素，则返回-1。 |
     | `ListIterator<E>` | `listIterator()`                                                                                                 返回列表中的列表迭代器（按适当的顺序）。 |
     | `ListIterator<E>` | `listIterator(int index)`                                                                                 从列表中的指定位置开始，返回列表中的元素（按正确顺序）的列表迭代器。 |
     | `E`               | `remove(int index)`                                                                                            删除该列表中指定位置的元素。 |
     | `boolean`         | `remove(Object o)`                                                                                               从列表中删除指定元素的第一个出现（如果存在）。 |
     | `boolean`         | `removeAll(Collection<?> c)`                                                                            从此列表中删除指定集合中包含的所有元素。 |
     | `boolean`         | `removeIf(Predicate<? super E> filter)`                                                删除满足给定谓词的此集合的所有元素。 |
     | `protected void`  | `removeRange(int fromIndex, int toIndex) `                                            从这个列表中删除所有索引在 `fromIndex` （含）和 `toIndex`之间的元素。 |
     | `void`            | `replaceAll(UnaryOperator<E> operator) `                                                将该列表的每个元素替换为将该运算符应用于该元素的结果。 |
     | `boolean`         | `retainAll(Collection<?> c) `                                                                       仅保留此列表中包含在指定集合中的元素。 |
     | `E`               | `set(int index, E element) `                                                                         用指定的元素替换此列表中指定位置的元素。 |
     | `int`             | `size()`                                                                                                                 返回此列表中的元素数。 |
     | `void`            | `sort(Comparator<? super E> c) `                                                                 使用提供的 `Comparator`对此列表进行排序以比较元素。 |
     | `Spliterator<E>`  | `spliterator()`                                                                                                  在此列表中的元素上创建*late-binding*和*故障快速* [`Spliterator`](https://www.matools.com/file/manual/jdk_api_1.8_google/java/util/Spliterator.html) 。 |
     | `List<E>`         | `subList(int fromIndex, int toIndex)`返回此列表中指定的 `fromIndex` （包括）和 `toIndex`之间的独占视图。 |
     | `Object[]`        | `toArray() `                                                                                                          以正确的顺序（从第一个到最后一个元素）返回一个包含此列表中所有元素的数组。 |
     | `<T> T[]`         | `toArray(T[] a) `                                                                                                以正确的顺序返回一个包含此列表中所有元素的数组（从第一个到最后一个元素）; 返回的数组的运行时类型是指定数组的运行时类型。 |
     | `void`            | `trimToSize()`                                                                                                    修改这个 `ArrayList`实例的容量是列表的当前大小。 |

     - [ ] 转换为数组                                           `toArray( )`

       ```java
       List<String> list = new ArrayList<String>();
       list.add("nihao");
       list.add("ma");   
       
       String[] array = new String[list.size()];
       array = list.toArray(array);
       
       // 或者
       String[] array = list.toArray(new String[list.size()]);
       ```

3. 常用方法 LinkedList：

   - [ ] 构造方法：同上

   - [ ] 常用方法：

     | Modifier and Type | Method and Description                                       |
     | ----------------- | ------------------------------------------------------------ |
     | `boolean`         | `add(E e)`将指定的元素追加到此列表的末尾。                   |
     | `void`            | `add(int index, E element)`在此列表中的指定位置插入指定的元素。 |
     | `boolean`         | `addAll(Collection<? extends E> c)`按照指定集合的迭代器返回的顺序将指定集合中的所有元素追加到此列表的末尾。 |
     | `boolean`         | `addAll(int index, Collection<? extends E> c)`将指定集合中的所有元素插入到此列表中，从指定的位置开始。 |
     | `void`            | `addFirst(E e)`在该列表开头插入指定的元素。                  |
     | `void`            | `addLast(E e)`将指定的元素追加到此列表的末尾。               |
     | `void`            | `clear()`从列表中删除所有元素。                              |
     | `Object`          | `clone()`返回此 `LinkedList`的浅版本。                       |
     | `boolean`         | `contains(Object o)`如果此列表包含指定的元素，则返回 `true` 。 |
     | `Iterator<E>`     | `descendingIterator()`以相反的顺序返回此deque中的元素的迭代器。 |
     | `E`               | `element()`检索但不删除此列表的头（第一个元素）。            |
     | `E`               | `get(int index)`返回此列表中指定位置的元素。                 |
     | `E`               | `getFirst()`返回此列表中的第一个元素。                       |
     | `E`               | `getLast()`返回此列表中的最后一个元素。                      |
     | `int`             | `indexOf(Object o)`返回此列表中指定元素的第一次出现的索引，如果此列表不包含元素，则返回-1。 |
     | `int`             | `lastIndexOf(Object o)`返回此列表中指定元素的最后一次出现的索引，如果此列表不包含元素，则返回-1。 |
     | `ListIterator<E>` | `listIterator(int index)`从列表中的指定位置开始，返回此列表中元素的列表迭代器（按适当的顺序）。 |
     | `boolean`         | `offer(E e)`将指定的元素添加为此列表的尾部（最后一个元素）。 |
     | `boolean`         | `offerFirst(E e)`在此列表的前面插入指定的元素。              |
     | `boolean`         | `offerLast(E e)`在该列表的末尾插入指定的元素。               |
     | `E`               | `peek()`检索但不删除此列表的头（第一个元素）。               |
     | `E`               | `peekFirst()`检索但不删除此列表的第一个元素，如果此列表为空，则返回 `null` 。 |
     | `E`               | `peekLast()`检索但不删除此列表的最后一个元素，如果此列表为空，则返回 `null` 。 |
     | `E`               | `poll()`检索并删除此列表的头（第一个元素）。                 |
     | `E`               | `pollFirst()`检索并删除此列表的第一个元素，如果此列表为空，则返回 `null` 。 |
     | `E`               | `pollLast()`检索并删除此列表的最后一个元素，如果此列表为空，则返回 `null` 。 |
     | `E`               | `pop()`从此列表表示的堆栈中弹出一个元素。                    |
     | `void`            | `push(E e)`将元素推送到由此列表表示的堆栈上。                |
     | `E`               | `remove()`检索并删除此列表的头（第一个元素）。               |
     | `E`               | `remove(int index)`删除该列表中指定位置的元素。              |
     | `boolean`         | `remove(Object o)`从列表中删除指定元素的第一个出现（如果存在）。 |
     | `E`               | `removeFirst()`从此列表中删除并返回第一个元素。              |
     | `boolean`         | `removeFirstOccurrence(Object o)`删除此列表中指定元素的第一个出现（从头到尾遍历列表时）。 |
     | `E`               | `removeLast()`从此列表中删除并返回最后一个元素。             |
     | `boolean`         | `removeLastOccurrence(Object o)`删除此列表中指定元素的最后一次出现（从头到尾遍历列表时）。 |
     | `E`               | `set(int index, E element)`用指定的元素替换此列表中指定位置的元素。 |
     | `int`             | `size()`返回此列表中的元素数。                               |
     | `Spliterator<E>`  | `spliterator()`在此列表中的元素上创建*late-binding*和*故障快速* [`Spliterator`](https://www.matools.com/file/manual/jdk_api_1.8_google/java/util/Spliterator.html) 。 |
     | `Object[]`        | `toArray()`以正确的顺序（从第一个到最后一个元素）返回一个包含此列表中所有元素的数组。 |
     | `<T> T[]`         | `toArray(T[] a)`以正确的顺序返回一个包含此列表中所有元素的数组（从第一个到最后一个元素）; 返回的数组的运行时类型是指定数组的运行时类型。 |

     - [ ] clone 浅版本  
     - [ ] offer  poll
     - [ ] push  pop
     - [ ] set 替换
     - [ ] 转换为数组

4. 常用方法 Vector：

   - [ ] 构造方法：

   - [ ] 常用方法：

     | Modifier and Type | Method and Description                                       |
     | ----------------- | ------------------------------------------------------------ |
     | `boolean`         | `add(E e)`将指定的元素追加到此Vector的末尾。                 |
     | `void`            | `add(int index, E element)`在此Vector中的指定位置插入指定的元素。 |
     | `boolean`         | `addAll(Collection<? extends E> c)`将指定集合中的所有元素追加到该向量的末尾，按照它们由指定集合的迭代器返回的顺序。 |
     | `boolean`         | `addAll(int index, Collection<? extends E> c)`将指定集合中的所有元素插入到此向量中的指定位置。 |
     | `void`            | `addElement(E obj)`将指定的组件添加到此向量的末尾，将其大小增加1。 |
     | `int`             | `capacity()`返回此向量的当前容量。                           |
     | `void`            | `clear()`从此Vector中删除所有元素。                          |
     | `Object`          | `clone()`返回此向量的克隆。                                  |
     | `boolean`         | `contains(Object o)`如果此向量包含指定的元素，则返回 `true` 。 |
     | `boolean`         | `containsAll(Collection<?> c)`如果此向量包含指定集合中的所有元素，则返回true。 |
     | `void`            | `copyInto(Object[] anArray)`将此向量的组件复制到指定的数组中。 |
     | `E`               | `elementAt(int index)`返回指定索引处的组件。                 |
     | `Enumeration<E>`  | `elements()`返回此向量的组件的枚举。                         |
     | `void`            | `ensureCapacity(int minCapacity)`如果需要，增加此向量的容量，以确保它可以至少保存最小容量参数指定的组件数。 |
     | `boolean`         | `equals(Object o)`将指定的对象与此向量进行比较以获得相等性。 |
     | `E`               | `firstElement()`返回此向量的第一个组件（索引号为 `0`的项目）。 |
     | `void`            | `forEach(Consumer<? super E> action)`对 `Iterable`的每个元素执行给定的操作，直到所有元素都被处理或动作引发异常。 |
     | `E`               | `get(int index)`返回此向量中指定位置的元素。                 |
     | `int`             | `hashCode()`返回此Vector的哈希码值。                         |
     | `int`             | `indexOf(Object o)`返回此向量中指定元素的第一次出现的索引，如果此向量不包含元素，则返回-1。 |
     | `int`             | `indexOf(Object o, int index)`返回此向量中指定元素的第一次出现的索引，从 `index`向前 `index` ，如果未找到该元素，则返回-1。 |
     | `void`            | `insertElementAt(E obj, int index)`在指定的index插入指定对象作为该向量中的一个 `index` 。 |
     | `boolean`         | `isEmpty()`测试此矢量是否没有组件。                          |
     | `Iterator<E>`     | `iterator()`以正确的顺序返回该列表中的元素的迭代器。         |
     | `E`               | `lastElement()`返回向量的最后一个组件。                      |
     | `int`             | `lastIndexOf(Object o)`返回此向量中指定元素的最后一次出现的索引，如果此向量不包含元素，则返回-1。 |
     | `int`             | `lastIndexOf(Object o, int index)`返回此向量中指定元素的最后一次出现的索引，从 `index` ，如果未找到元素，则返回-1。 |
     | `ListIterator<E>` | `listIterator()`返回列表中的列表迭代器（按适当的顺序）。     |
     | `ListIterator<E>` | `listIterator(int index)`从列表中的指定位置开始，返回列表中的元素（按正确顺序）的列表迭代器。 |
     | `E`               | `remove(int index)`删除此向量中指定位置的元素。              |
     | `boolean`         | `remove(Object o)`删除此向量中指定元素的第一个出现如果Vector不包含元素，则它不会更改。 |
     | `boolean`         | `removeAll(Collection<?> c)`从此Vector中删除指定集合中包含的所有元素。 |
     | `void`            | `removeAllElements()`从该向量中删除所有组件，并将其大小设置为零。 |
     | `boolean`         | `removeElement(Object obj)`从此向量中删除参数的第一个（最低索引）出现次数。 |
     | `void`            | `removeElementAt(int index)`删除指定索引处的组件。           |
     | `boolean`         | `removeIf(Predicate<? super E> filter)`删除满足给定谓词的此集合的所有元素。 |
     | `protected void`  | `removeRange(int fromIndex, int toIndex)`从此列表中删除所有索引为 `fromIndex` （含）和 `toIndex`之间的元素。 |
     | `void`            | `replaceAll(UnaryOperator<E> operator)`将该列表的每个元素替换为将该运算符应用于该元素的结果。 |
     | `boolean`         | `retainAll(Collection<?> c)`仅保留此向量中包含在指定集合中的元素。 |
     | `E`               | `set(int index, E element)`用指定的元素替换此Vector中指定位置的元素。 |
     | `void`            | `setElementAt(E obj, int index)`设置在指定的组件 `index`此向量的要指定的对象。 |
     | `void`            | `setSize(int newSize)`设置此向量的大小。                     |
     | `int`             | `size()`返回此向量中的组件数。                               |
     | `void`            | `sort(Comparator<? super E> c)`使用提供的 `Comparator`对此列表进行排序以比较元素。 |
     | `Spliterator<E>`  | `spliterator()`在此列表中的元素上创建*late-binding*和*故障切换* [`Spliterator`](https://www.matools.com/file/manual/jdk_api_1.8_google/java/util/Spliterator.html) 。 |
     | `List<E>`         | `subList(int fromIndex, int toIndex)`返回此列表之间的fromIndex（包括）和toIndex之间的独占视图。 |
     | `Object[]`        | `toArray()`以正确的顺序返回一个包含此Vector中所有元素的数组。 |
     | `<T> T[]`         | `toArray(T[] a)`以正确的顺序返回一个包含此Vector中所有元素的数组; 返回的数组的运行时类型是指定数组的运行时类型。 |
     | `String`          | `toString()`返回此Vector的字符串表示形式，其中包含每个元素的String表示形式。 |
     | `void`            | `trimToSize()`修改该向量的容量成为向量的当前大小。           |

     - [ ] toString


