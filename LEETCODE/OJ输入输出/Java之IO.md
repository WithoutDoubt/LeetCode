第一步是 输入输出

关键点是：数字和字符串

其中 ： `cin.nextLine()`  可能是空白行，因为 `nextLine() 读取到 ”\r“ 就结束了，"\r"的意思是 Enter换行` 

```java
// 关于牛客的IO输入10道题
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import static java.util.Arrays.sort;


// 研究IO输入输出

public class Main {
    public static void main(String[] args) {
//        functionTwo();
        functionFour();
    }

    // 第一种写法：
    private static void functionOne()  {
        Scanner cin = new Scanner(System.in);
        int a, b;
        while(cin.hasNext()){
            String line = cin.nextLine();
            String[] strings = line.split(" ");
            a = Integer.valueOf(strings[0]);
            b = Integer.valueOf(strings[1]);
            System.out.println(a+b);
        }
    }

    // 第二种写法；
    private static void functionTwo(){
        Scanner cin = new Scanner(System.in);
        int a,b;
        int n = cin.nextInt();


        while(n-- > 0){
            a = cin.nextInt();
            b = cin.nextInt();
            System.out.println(a+b);
        }
    }

    // 第三种写法：
    private static void functionThree(){
        Scanner cin = new Scanner(System.in);
        int a,b;
        int n = cin.nextInt();


        while(n-- > 0){
            a = cin.nextInt();
            b = cin.nextInt();
            System.out.println(a+b);
        }
    }
    // 第四种写法：
    private static void functionFour(){
        Scanner cin = new Scanner(System.in);
        int a, b , sum = 0;
        while (cin.hasNext()){
            int tag = cin.nextInt();
            if(tag == 0) return;
            for (int i = 0; i < tag; i++) {
                sum += cin.nextInt();
            }
            System.out.println(sum);
            sum = 0;
        }

    }
    // 第五种写法
    private static void functionFive(){
        Scanner cin = new Scanner(System.in);
        int n = cin.nextInt();
        cin.nextLine();              // 读取多余的  "/r"
        for (int i = 0; i < n; i++) {
            String[] a = cin.nextLine().split(" ");
            int sum = 0;
            for (int j = 1; j < a.length; j++) {
                sum += Integer.valueOf(a[j]);

            }

            System.out.println(sum);
        }
    }


    // 第六种写法：

    private static void functionSix(){
        Scanner cin = new Scanner(System.in);
        while (cin.hasNextLine()){
            int sum = 0;
            String[] a = cin.nextLine().split(" ");
            for (int i = 1; i < a.length; i++) {
                sum += Integer.valueOf(a[i]);
            }
            System.out.println(sum);
        }
    }

    // 第七种写法：

    private static void functionSeven(){

        Scanner cin = new Scanner(System.in);

        while(cin.hasNextLine()){
            int sum = 0;                     // 每一次循环完更新一次sum
            String[] a = cin.nextLine().split(" ");
            for(int i = 0; i < a.length;i++){
                sum  += Integer.valueOf(a[i]);
            }
            System.out.println(sum);
        }
    }

    // 字符串第一题
    private static void functionCharOne(){
        Scanner cin = new Scanner(System.in);
        int n = cin.nextInt();
        cin.nextLine();
        String[] a = cin.nextLine().split(" ");
        Arrays.sort(a);
        for (int i = 0; i < a.length; i++) {
            if(i == a.length-1){
                System.out.println(a[i]);
            }else {
                System.out.print(a[i] + " " +
                        "");
            }
        }


    }


    // 字符串第二题
    private static void functionCharTwo(){
        Scanner cin = new Scanner(System.in);
        while (cin.hasNextLine()){
            String[] a = cin.nextLine().split(" ");
            Arrays.sort(a);
            for (int i = 0; i < a.length; i++){
                if (i == a.length -1){
                    System.out.println(a[i]);
                }else {
                    System.out.print(a[i]+" ");
                }
            }
        }
    }

    // 字符串第三题
    private static void functionCharThree(){
        Scanner cin = new Scanner(System.in);

        while(cin.hasNextLine()){
            String[] a = cin.nextLine().split(",");
            Arrays.sort(a);
            for (int i = 0; i < a.length; i++) {
                if(i == a.length-1){
                    System.out.println(a[i]);  // 最后一行有换行符
                }else {
                    System.out.print(a[i]);
                    System.out.print(",");
                }
            }
        }
    }


}

```

-  `valueOf` 和 `parseInt` 有什么区别？

  `valueOf` 返回的是包装类型，`parseInt`是基本类型

  都是Integer类中的方法

- 

