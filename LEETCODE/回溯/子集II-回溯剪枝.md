- [ ] <https://leetcode-cn.com/problems/subsets-ii/>

```c++
class Solution{
public:
    vector<vector<int> > res;
    vector<vector<int> > subsetsWithDup(vector<int>& nums){
        vector<int> tmp;
        sort(nums.begin(),nums.end());
        backtrack(nums,0,tmp);
        return res;
    }
    
    void backtrack(vector<int>& nums, int start, vector<int>& tmp){
        res.push_back(tmp);
        for (int i = start, i < nums.size(); i++){
            
            if (i > start && nums[i-1] == nums[i])  //去重
                continue;
            
            tmp.push_back(nums[i]);
            bactrack(nums,i+1,tmp);
            tmp.pop_back();
        }
    }
}
```

