<https://leetcode-cn.com/problems/top-k-frequent-elements/>



```c++
class Solution {
    // 求前 k 大，用小根堆，求前 k 小，用大根堆。

    /*
    1. topk （前k大）用小根堆，维护堆大小不超过 k 即可。每次压入堆前和堆顶元素比较，
    如果比堆顶元素还小，直接扔掉，否则压入堆。检查堆大小是否超过 k，如果超过，弹出堆顶。
    复杂度是 nlogk
    2. 避免使用大根堆，因为你得把所有元素压入堆，复杂度是 nlogn，
    而且还浪费内存。如果是海量元素，那就挂了。
    */
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int,int> map;

        for (int i : nums) map[i]++;

        // q 是小顶堆
        priority_queue< pair<int,int>, vector<pair<int,int> >, greater<pair<int,int> > > q;

        for (auto it : map){
            if (q.size() == k){ // 维持k个
                if (it.second > q.top().first){  
                    q.pop();
                    q.push(make_pair(it.second,it.first));
                }
            }
            else {  // 存入键值对, pair(频率,值)
                q.push(make_pair(it.second,it.first));
            }
        }
        vector<int> res;
            while(q.size()){
                res.push_back(q.top().second);
                q.pop();
            }
        return vector<int> (res.rbegin(),res.rend());
    }
};
```

