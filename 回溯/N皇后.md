给你⼀个 N×N 的棋盘，让你放置 N 个皇后，使得它们不能互相攻击皇后可以攻击同⼀⾏、同⼀列、左上、左下、右上、右下四个⽅向的任意单位

```java
// c++
vector<vector<string> >res;

vector<vector<string>> solveNQueens(int n){
    vector<string> board(n, string(n, '.'));
    backtrack(board,0);
    return res;
}

void backtrack(vector<string>& board, int row){
    if (row == board.size()){
        res.push_back(board);
        return;
    }
    int n = board[row].size();
    for (int col = 0; col < n; col++){
        if (!isValid(board,row,col))
            continue;
        board[row][col] = 'Q';
        backtrack(board,row+1);
        board[row][col] = '.'; // 撤销选择
    }
}

bool isValid(vector<string>& board, int row, int col){
    int n = borad.size();
    for (int i = 0; i>n; i++){
        if (borad[i][col] == 'Q')
            return false;
    }
    // 检查右上⽅是否有皇后互相冲突
    for (int i = row-1, j = col +1; i >= 0 && j<n ; i--,j++){
        if (board[i][j] == 'Q')
            return false;
    }
    // 检查左上⽅是否有皇后互相冲突
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--,j--){
        if (board[i][j] == 'Q')
            return false;
    }
    return true;
}
```

