```c++
bool isValidBSF(TreeNode* root){
    if (root == NULL) return true;
    
    if (root->left != NULL && root->val <= root->left->val) return false;
    if (root->right != NULL && root->val >= root->right->val) return false;
    
    return isValidBST(root->left)&&isValidBST(root->right);
}
```





```c++
bool isValidBST(TreeNode* root){
    return isValidBST(root, NULL, NULL);
} // 需要辅助函数

bool isValidBST(TreeNode* root, TreeNode* min, TreeNode* max){
    if (root == NULL) return true;
    if (min != NULL && root->val <= min->val) return false;
    if (max != NULL && root->val >= max->val) return false;
    
    return isValidBSF(root->left, min, root)
        && isValidBSF(root->right, root, max);
}
```

