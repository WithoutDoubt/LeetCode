#### 使用map记录

```c++
class Solution{
public:
	unordered_map<int,int> m;
	vector<vector<int>> res;
	vector<int> track;
	
	vector<vector<int>> permuteUnique(vector<int>& nums){
        sort(nums.begin(),nums.end());
        for (int c : nums){
            m[c]++;
        } 
        backtrack(nums); 
        return res;   
    }   
	
    void backtrack(vector<int>& nums){
        if (track.size() == nums.size()){
            res.push_back(track);
            return;
        }
        for(auto &item : m){  // 循环访问 map中的元素，自带去重
            if (item.second == 0) continue;
            item.second--;
            track.push_back(item.first);
            backtrack(nums);
            item.second++;
            track.pop_back();
        }
        
    }

    
};

// 为了加快速度，将nums,size()存为全局变量

class Solution{
public:
	unordered_map<int,int> m;
	vector<vector<int>> res;
	vector<int> track;
    int len;
	
	vector<vector<int>> permuteUnique(vector<int>& nums){
        len = nums.size();
        for (int c : nums){
            m[c]++;
        } 
        backtrack(); 
        return res;   
    }   
	
    void backtrack(){
        if (track.size() == len){
            res.push_back(track);
            return;
        }
        for(auto &item : m){  // 循环访问 map中的元素，自带去重
            if (item.second == 0) continue;
            item.second--;
            track.push_back(item.first);
            backtrack();
            item.second++;
            track.pop_back();
        }
        
    }

    
};
```





```c++
class Solution {
    vector<int> nums;//记录每一次选择后数组状态，包括最终答案
    vector<vector<int>> ans;//选择完每一个数组后并入答案集
    int len;//输入数组元素数量
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        len=nums.size();//更新len
        this->nums=nums;//更新nums
        dfs(0);//开始回溯
        return ans;
    }
    void dfs(int n){
        if(n==len){
            ans.push_back(nums);//已经排列到len位置，即超出数组范围，这意味着已经完成了排列，将此排列并入答案集合
            return;
        }
        vector<int> temp={};//记录该位选择过的元素值，已经选择过的值不再选择
        for(int i=n;i<len;++i){//n为当前正在选择的位，i为准备要作为n位元素目前的位置
            if(find(temp.begin(),temp.end(),nums[i])!=temp.end())continue;//已经选择过的值不再选择
            swap(nums[n],nums[i]);//将第i位数字移动到n位，完成该位选择
            temp.push_back(nums[n]);//记录选择，防止选择相等数字产生多余的解
            dfs(n+1);//选择下一位数字
            swap(nums[n],nums[i]);//变为选择之前的状态，重新选择下一位数字
        }
    }
};

```



`next_permutation: 全排列函数`

```c++
class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        result.push_back(nums);
        while(next_permutation(nums.begin(), nums.end())) {
            result.push_back(nums);
        }
        return result;
    }
};

```

自己实现：

```c++
class Solution {
public:
    bool nextPermutation(vector<int>& nums) {
        auto i = is_sorted_until(nums.rbegin(), nums.rend()); 
        // 找到末尾的一个降序段[s]及其前一个元素i
        
        bool has_next = i != nums.rend();
        if(has_next) {
            iter_swap(i, upper_bound(nums.rbegin(), i, *i));  // 找到[s]中比i大的数中最小的
            reverse(nums.rbegin(), i);                        // 序列反转
        }
        return has_next;
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> result;
        sort(nums.begin(), nums.end());
        result.push_back(nums);
        while(nextPermutation(nums)) {
            result.push_back(nums);
        }
        return result;
    }
};

```

