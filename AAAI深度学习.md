Q = K=V 都是(n,m)矩阵 ，n为`seq_len`，$(x_1,x_2,...,x_n)^T​$   
$$
Q^T*K =
\begin{bmatrix}
a_0\quad a_1\\b_0\quad b_1\\c_0\quad c_1
\end{bmatrix}
*
\begin{bmatrix}
a_0  \quad b_0\quad c_0\\
a_1\quad b_1\quad c_1
\end{bmatrix}
= 
\begin{bmatrix}
aa,ab,ac\\
ba,bb,bc\\
ca,cb,cc
\end{bmatrix}
$$
保留下三角
$$
(Q^T*K)*V= (m,n)(n,m)*(m*n) = 

\begin{bmatrix}
aa,ab,ac\\
ba,bb,bc\\
ca,cb,cc
\end{bmatrix} *
\begin{bmatrix}
a_0\quad a_1\\b_0\quad b_1\\c_0\quad c_1
\end{bmatrix}
=
\begin{bmatrix}
a*aa + b*ab + c*ac \\
a*ba + b*bb + c*bc \\
a*ac + b*bc + c*cc
\end{bmatrix}
$$
现在可以看出来 第一个值 只和 `aa`有关， 第二个值只和`ab,bb`有关，第三个值和大家都有关
$$
\begin{bmatrix}
aa,0,0\\
ba,bb,0\\
ca,cb,cc
\end{bmatrix} *
\begin{bmatrix}
a\\
b\\
c\\
\end{bmatrix}=
\begin{bmatrix}
a*aa + b*0 + c*0 \\
a*ba + b*bb + c*0 \\
a*ac + b*bc + c*cc
\end{bmatrix} 
= 
\begin{bmatrix}
a*aa  \\
a*ba + b*bb  \\
a*ac + b*bc + c*cc
\end{bmatrix} 
$$
所以 `seq_len = n`

d_model 为 input 就是 特征维度  C

n_words 为 output 就是 输出维度 C'



