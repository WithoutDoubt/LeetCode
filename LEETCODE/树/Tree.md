1. 平衡二叉树 110

   - 首先如何判断一个结点？
   - 就是判断该结点左、右子树是否是平衡的，如果是，返回高度；不是，返回-1；
   - 如果有-1，就一定不是平衡的，所以是一个后序遍历的过程；

   ```java
   /**
    
   */
   class Solution {
       public boolean isBalanced(TreeNode root) {
           /*
               判断每个子节点的左右高度
               如何计算左右高度, 然后，遍历左右子树节点
   
           */
           
           return getHeight(root) == -1 ? false :true;
       }
   
       private int getHeight(TreeNode root){
           if (root == null) return 0;
   
           int h_l = getHeight(root.left);
           int h_r = getHeight(root.right);
   
           if(h_l < 0 || h_r < 0) return -1;
           if(Math.abs(h_l - h_r) > 1) return -1;
           
   
           return Math.max(h_l,h_r)+1 ;
       }
   }
   ```

2. 特定深度结点链表 

   - 就是层次遍历，只不过每一层的节点存在单独的数组中；所以创建一个List
   - 层次遍历时，如何知道每一层结束了？在保存节点到队列的时候，用一个标志来标记一下
   - 逻辑上：一开始 `1，null -> queue`, 然后pop节点，操作节点，再判断是否有左右节点，如果有，存入queue，如果pop的节点是null，那么说明上一层的节点都结束了，也就说明下一层的节点都读入了，可以存入 标志null了，从而达到记录层数的目的。
   - **或者用size 记录当前队列中 节点 数目**

   ```java
   class Solution{
       public ListNode[] listOfDepth(TreeNode tree){
           LinkedList<TreeNode> queue = new LinkedList<>();
           queue.offer(tree);
           
           List<ListNode> res = new ArrayList<>();
           ListNode dummy = new ListNode(0);
           
           while(!queue.isEmpty()){
               int size = queue.size();                    // 当前节点数目
               ListNode curr = dummy;
               
               for (int i = 0; i < size; i++){
               	TreeNode treeNode = queue.poll();  // 吐出结点
                   
                   curr.next = new ListNode(treeNode.val);
                   if (treeNode.left != null){
                       queue.offer(treeNode.left);
                   }
                   if (treeNode.right != null){
                       queue.offer(treeNode.right);
                   }
                   curr = curr.next;
               }
               
               res.add(dummy.next);
               dummy.next = null;
           }
           return res.toArray(new ListNode[] {});
       }
   }
   ```

3. 

