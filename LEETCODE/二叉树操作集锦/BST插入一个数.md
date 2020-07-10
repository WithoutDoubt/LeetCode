```c++
TreeNode inseretIntoBST(TreeNode root, int val){
    if (root == NULL) return new TreeNode(val);
    if (root->val < val) root->right = insertIntoBST(root->right,val);
    if (root->val > val) root->right = insertIntoBST(root->left,val);
    return root;
    
}
```

