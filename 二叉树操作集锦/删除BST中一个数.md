```c++
TreeNode* deleteNode(TreeNode* root, int key){
    if (root == NULL) return NULL;
    
    if (root->val == key){
    	if (root->left == NULL) return root->right;
        if (root->right == NULL) return root->left;
        // 有左右子树
        TreeNode* minNode = getMin(root->right);
        root->val = minNode->val;
        root->right = deleteNode(root->right,minData->val);
    }
    else if (root->val > key) root->left = deleteNode(root->left, key);
    else if (root->val < key) root->left = deleteNode(root->right, key);
    
    return root;
    
}

TreeNode* getMin (TreeNode* node){
    while (node->left != NULL) 
        node = node->left;
    return node;
}
```

