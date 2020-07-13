<https://leetcode-cn.com/problems/jump-game-ii/>



#### 贪心

```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        int maxPos = 0, n = nums.size(), end = 0, step = 0;
        
        for (int i = 0; i < n - 1; ++i) {  // n-1 表示倒数第二个
            if (maxPos >= i) {             // 如果max >= i 表示可以到
                maxPos = max(maxPos, i + nums[i]);  //
                if (i == end) {          // 到达边界，就加1
                    end = maxPos;
                    ++step;
                }
            }
        }
        return step;
    }
};
```

