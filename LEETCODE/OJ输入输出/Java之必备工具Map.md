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

#### Map

1. 理论上：

   - [ ] Java.util.Map

   - [ ] Map 接口下三个实现类：HashMap、TreeMap、HashTable

     - [ ] HashMap:

       > 无序的，不同步，运行null值

     - [ ] HashTable：

       > 无序的，同步的，不允许null值

     - [ ] TreeMap：

       > 有序的（内部排序），

2. 常用方法 HashMap：

   - [ ] 构造方法：

     > `Map<Integer,Integer> map = new HashMap<>();`

   - [ ] 常用方法：增删改查

     - [ ] | Modifier and Type     | Method and Description                                       |
       | --------------------- | ------------------------------------------------------------ |
       | `void`                | `clear()`从该地图中删除所有的映射（可选操作）。              |
       | `default V`           | `compute(K key, BiFunction<? super K,? super V,? extends V> remappingFunction)`尝试计算指定键的映射及其当前映射的值（如果没有当前映射， `null` ）。 |
       | `default V`           | `computeIfAbsent(K key, Function<? super K,? extends V> mappingFunction)`如果指定的键尚未与值相关联（或映射到 `null` ），则尝试使用给定的映射函数计算其值，并将其输入到此映射中，除非 `null` 。 |
       | `default V`           | `computeIfPresent(K key, BiFunction<? super K,? super V,? extends V> remappingFunction)`如果指定的密钥的值存在且非空，则尝试计算给定密钥及其当前映射值的新映射。 |
       | `boolean`             | `containsKey(Object key)`如果此映射包含指定键的映射，则返回 `true` 。 |
       | `boolean`             | `containsValue(Object value)`如果此地图将一个或多个键映射到指定的值，则返回 `true` 。 |
       | `Set<Map.Entry<K,V>>` | `entrySet()`返回此地图中包含的映射的[`Set`](https://www.matools.com/file/manual/jdk_api_1.8_google/java/util/Set.html)视图。 |
       | `boolean`             | `equals(Object o)`将指定的对象与此映射进行比较以获得相等性。 |
       | `default void`        | `forEach(BiConsumer<? super K,? super V> action)`对此映射中的每个条目执行给定的操作，直到所有条目都被处理或操作引发异常。 |
       | `V`                   | `get(Object key)`返回到指定键所映射的值，或 `null`如果此映射包含该键的映射。 |
       | `default V`           | `getOrDefault(Object key, V defaultValue)`返回到指定键所映射的值，或 `defaultValue`如果此映射包含该键的映射。 |
       | `int`                 | `hashCode()`返回此地图的哈希码值。                           |
       | `boolean`             | `isEmpty()`如果此地图不包含键值映射，则返回 `true` 。        |
       | `Set<K>`              | `keySet()`返回此地图中包含的键的[`Set`](https://www.matools.com/file/manual/jdk_api_1.8_google/java/util/Set.html)视图。 |
       | `default V`           | `merge(K key, V value, BiFunction<? super V,? super V,? extends V> remappingFunction)`如果指定的键尚未与值相关联或与null相关联，则将其与给定的非空值相关联。 |
       | `V`                   | `put(K key, V value)`将指定的值与该映射中的指定键相关联（可选操作）。 |
       | `void`                | `putAll(Map<? extends K,? extends V> m)`将指定地图的所有映射复制到此映射（可选操作）。 |
       | `default V`           | `putIfAbsent(K key, V value)`如果指定的键尚未与某个值相关联（或映射到 `null` ）将其与给定值相关联并返回 `null` ，否则返回当前值。 |
       | `V`                   | `remove(Object key)`如果存在（从可选的操作），从该地图中删除一个键的映射。 |
       | `default boolean`     | `remove(Object key, Object value)`仅当指定的密钥当前映射到指定的值时删除该条目。 |
       | `default V`           | `replace(K key, V value)`只有当目标映射到某个值时，才能替换指定键的条目。 |
       | `default boolean`     | `replace(K key, V oldValue, V newValue)`仅当当前映射到指定的值时，才能替换指定键的条目。 |
       | `default void`        | `replaceAll(BiFunction<? super K,? super V,? extends V> function)`将每个条目的值替换为对该条目调用给定函数的结果，直到所有条目都被处理或该函数抛出异常。 |
       | `int`                 | `size()`返回此地图中键值映射的数量。                         |
       | `Collection<V>`       | `values()`返回此地图中包含的值的[`Collection`](https://www.matools.com/file/manual/jdk_api_1.8_google/java/util/Collection.html)视图。 |

     - [ ] 修改value值就是重新put                                           

