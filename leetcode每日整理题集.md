##### 1.给定一个非负整数 N，找出小于或等于 N 的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增。

（当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。）

链接：https://leetcode-cn.com/problems/monotone-increasing-digits
1.对于这个问题我做了两种解法，一种是非常简单的想法通过对于从给定的N开始通过将N转化为字符串再化为字符数组，然后排序后在比较与原来的字符串是否相同(有点绕，毕竟是我这个算法新人想出来的而且还通不过当前给的较大的事例。。。。。。，下面我代码的时间复杂度为O(N)(N指的是当前给定的数字N）

```java
public int monotoneIncreasingDigits(int N) {
      
          String snew=new String();
        while(N>=0){
            String s = String.valueOf(N);
            char [] c=s.toCharArray();
            Arrays.sort(c);
            snew=String.valueOf(c);
            if(snew.equals(s)) break;
            N-=1;
        }

        return Integer.parseInt(snew);
        

    }
```

第二种当然是参考一位大哥的，它通过将当前的数字->字符串->数组，然后倒序比较当前位置与相邻左边位置的数字的大小，如果当前位置i小于其相邻左边位置的我们设置标记flag=i，并且将c[i-1]--，对于整个位置循环比较完成后我们开始将标记位置flag一直到字符数组尾部都转化为9（要求求得是最接近的数，所以我们这样转换），这样整段代码的时间复杂度就转化为O(n)（在这里我们的复杂度值得是该数字的长度，就比我之前写的代码好多了，

```java
public class Solution {
    public int monotoneIncreasingDigits(int N) {
            String s = String.valueOf(N);
            char [] c=s.toCharArray();
            int flag=c.length;
            for(int i=c.length-1;i>0;i--){
                if(c[i]<c[i-1]){
                    c[i-1]--;
                    flag=i;
                }
            }
            for(int i=flag;i<c.length;i++){
                c[i]='9';
            }
        return Integer.parseInt(new String(c));
        

    }
}

```

2.12月16号

给定一种规律 pattern 和一个字符串 str ，判断 str 是否遵循相同的规律。

这里的 遵循 指完全匹配，例如， pattern 里的每个字母和字符串 str 中的每个非空单词之间存在着双向连接的对应规律。

如 abba 对应"dog cat cat dog"，那么我们认为该字符串符合abba规律，该题目顾名思义就是判断字符串str是否符合str，这题我做了一种解法(参考他人的，我自己没想出来)，要写出这道题，我们需要考虑整道题的具体关系，在求解abba 对于"dog cat cat dog"的情况中，我们其实可以吧问题拆解为对于字符'a'必须与"dog"对应，那么我们可以利用字典的特性来解决这个问题，就创立key值为字符的，value值为字符串的字典，将字符与对应字符串绑定起来，之后

1.在每次循环遍历检查字典中是否含有该键值

2.如果含有，那么我们取出该键值对应的字符串判断是否与当前字符串相等，不相等返回错误

2.如果没有，我们判断字典是否含有该value值，如果有说明当前出现对应错误，我们返回错误值，否则添加key 与value值

具体代码：

```java
  public boolean wordPattern(String pattern, String s) {
            Map <Character,String> m=new HashMap<>();
         
            String [] s1=s.split(" ");
            if(pattern.length()!=s1.length) return false;
            for(int i=0;i<pattern.length();i++){
                if(!m.containsKey(pattern.charAt(i))){
                    if(m.containsValue(s1[i])) return false;
                    m.put(pattern.charAt(i),s1[i]);
                }
                else{
                    if(!m.get(pattern.charAt(i)).equals(s1[i])) return false;
                }
            }
            return true;
        }

```

#### [714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee
对于该题目我们分析，在每天的每次抉择中我们需要决定是买还是抛售，那么这就相当于每天我们都需要做一个决定，即买股票-price，or买入那么对于这样的递推公式我们可以利用动态规划法，对于该问题做如下分析

1.当我们手里有股票，我们只能抛售，或不卖，当我们手里没有股票，我们可以买或者不买，我们每次在每阶段希望获得最大收益，那么在对于第i天我们能获得最大的价值就是 value[i][0]代表我们不买钞票那就比较抛售比较大，还是不买股票价值比较大，那么我们手中拥有股票就要比较卖了价值比较大，还是不卖较大

具体代码：

```java
    public int maxProfit(int[] prices, int fee) {
        int len = prices.length;
        int [][] dp=new int[len][2];
        dp[0][0]=0;
        dp[0][1]=-prices[0];
        for(int i=1; i<len; i++){
            dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]-fee);
            dp[i][1]=Math.max(dp[i-1][0]-prices[i],dp[i-1][1]);
        }
        return dp[length-1][0];

    }
```

但是在使用数组会达到6ms的时间，我们其实还有优化的空间，我们发现，我们最后想要的结果只需要用两个值去推出他来，所以我们可以做如下优化，减少空间复杂度的同时，创立普通变量的时间也会比数组减小2ms（leetcode的测试通过下面这段代码只需4ms)具体代码如下

```java
public int maxProfit(int[] prices, int fee) {
        if(prices == null||prices.length<2) return 0;
        int len = prices.length;
        int hold,noHold;
      noHold=0;
      hold=-prices[0];
      
        for (int i = 1; i < len; i++) {
           
            noHold = Math.max(noHold, hold + prices[i] - fee);
      hold = Math.max(hold, noHold - prices[i]);
        }
        return noHold;

    }
```

#### [389. 找不同](https://leetcode-cn.com/problems/find-the-difference/)

给定两个字符串 ***s*** 和 ***t***，它们只包含小写字母。

字符串 t由字符串 s随机重排，然后在随机位置添加一个字母。

请找出在 ***t*** 中被添加的字母。

 对于给定的字符串s,以及任意位置插入一个字母的新字符串t，要寻找添加的字母，对于这个任务看似很难，因为要顾及这是随机插入的位置，以及t是s的随机重排，看似用蛮力法较难获得需要找到规律，所以我们换种思想来比较，首先我们如果对这两个字符串进行排序（转化为字符数组后）,那么新增加的字母只需比较排序后的哪位不同即可（排序后，例如abcd aebcd排序后，abcd abcde，那么我们只需要比较以下就会发现新增加的是'e'，按照这个思路我们编写代码：

```java
    public char findTheDifference(String s, String t) {
        char [] str1=s.toCharArray();
        char [] str2=t.toCharArray();
        if(s.length()<1) return str2[0];
        Arrays.sort(str1);
        Arrays.sort(str2);
        int i;
        for(i=0;i<str1.length;i++) {
        	if(str1[i]!=str2[i]) {return str2[i];
        	}
}      
     return str2[i];
    }

```

这段代码再具体执行的时候时间复杂度为O(N),但因为排序算法耽误了点时间，所以leetcode在测试后反馈输出为3ms，相对于现在大部分人解法，算比较慢的。

​		所以在参考别人写的代码后，我们解决问题其实可以这么想，即将插入前的字符串与插入后的字符串连接，如果插入后的字母为a，则a的个数必为奇数，对于这个规律我们可以使用位运算的异或运算来进行运算，对于为偶数的字母异或运算，则结果为0,对于为奇数的字母异或运算，结果等于它本身，这样我们对于上面的一种解法，我们不需要排序来占用额外的时间，这份代码的时间复杂度仍然为O(n)，但由于少了排序，它在leetcode上所需时间1ms，代码如下：

```Java
public char findTheDifference(String s, String t) {
        char[] charArr = s.concat(t).toCharArray();
        char res = 0;
        for (char c : charArr) {
            res ^= c;
  
        }
        return res;
    }

```

#### [746. 使用最小花费爬楼梯](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)

难度：简单

数组的每个索引作为一个阶梯，第 `i`个阶梯对应着一个非负数的体力花费值 `cost[i]`(索引从0开始)。

每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。

您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。

​	我对于这道题的分析：对于这种题目，多阶段规划并求最优值的方法，我们可以选择动态规划法这种方法，对于爬到第i层阶梯的花费，我们设置动态规划方程为dp[i]=min(dp[i-1]+cost[i],dp[i-2]+cost[i])，那么我们整个问题迎刃而解.

​	但是在完成代码后提交得时候，我发现这道题题目藏了一个坑，就是我们的最终目的地不是最后一层阶梯而是阶梯之上的楼顶，那么也就是我们可以从倒数第二层阶梯直接走过去，也可以从最后一层阶梯上去（注意如果这样的话他们是默认不需要消耗体力的，所以我们还要在设置最后一层比较dp[len-1],和dp[i-2]谁比较小），这样代码才算可以，该代码时间复杂度为O(N)，空间复杂度也为O(N),具体代码如下：

```java
	public int min(int a,int b) {
		return a<b?a:b;
	}
    public int minCostClimbingStairs(int[] cost) {
    	
    	int len=cost.length;
    	if(len<3) return cost[len-1];
    	int [] res =new int[len];
    	res[0]=cost[0];
    	res[1]=cost[1];
    	for(int i=2;i<len;i++) {
    		res[i]=min(res[i-1]+cost[i],res[i-2]+cost[i]);
    		
    		
    	}
        res[len-1]=min(res[len-1],res[len-2]);
    	return res[len-1];

    }
```

#### [103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

难度中等345

给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

问题分析：对于这样的问题，需要判断每次遍历的层数，比如从第一层（节点不算）开始，是从左到右，而下一层有时从右到左，那么我们想对于最终返回结果：List<List<Integer>> list，我们不要局限于到底是从左往右添加还是从右往左添加值，我们换个方向想对于list的插入来说从左往右是不是相当于左边节点的值在右边节点的前面，那么我们按照一般的插入就可以解决，那么如果换一个右边的节点在左边的节点前面要怎么取舍呢，如果这时候我们还是采用先插入左边但我们在头部插入是不是就可以做到右边的在左边的前面，那么我们就可以利用相同的插入顺序，但是插入位置的不同来解决这个问题了，这样做出来的代码时间复杂度为O(N)（N为节点的个数），代码如下 ：

```java
public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
    int	flag=1;
    List<List<Integer>> list=new ArrayList<>();
    	dfs(root,list,0);
    	
    return list	;
    }
    public void dfs(TreeNode root,List<List<Integer>> list,int level) {
    	if(root==null) {
    		return;
    	}
    	if(list.size()==level) {
    		list.add(new ArrayList<Integer>());
    	}
    	if((level&1)==1) {
    		list.get(level).add(0,root.val);
    	}
    	else {
    		list.get(level).add(root.val);
    	}
    	dfs(root.left,list,level+1);
    	dfs(root.right,list,level+1);
    }
//12月22号 18时55分写
```

#### [66. 加一](https://leetcode-cn.com/problems/plus-one/)

难度简单602

给定一个由 **整数** 组成的 **非空** 数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储**单个**数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。

问题解决：对于该题貌似是一个进位算法的问题，想都不用想他给我们的测试上一定会有什么，99999的进位，我们唯一要特别注意处理的就是这种特殊情况，我们想象怎么处理呢，观察题目可知，他每次只会加1，那么如果原位进位就会变成0，比如399 -> 400那么我们只需要在做的时候先判断加和的时候会不会等于10（由于只加一，所以他只有产生10才能进位，所以我们注意一下，就好了当等于10时原位取0，然后我们向前面一位继续加一当产生比如9999 ->10000这种操作的话，我们换种思维想想结果只不过是把原有数组长度加一然后把第一个值变为1，那么我们只需要再设置判断就可以完成我们的结果，

这样写出来的代码时间复杂度为O(N)，N为数组的长度，运行0ms 自然run beats 100%，代码如下：

```java
    public int[] plusOne(int[] digits) {
    	int len=digits.length-1;
    	int i,j,k;
    	int flag=1;
    	do {
    	k=digits[len]+flag;
    	if(k<10) {
    		flag=0;
    		digits[len]=k;
    	}
    	else {
    		flag=1;
    		digits[len]=0;
            len--;
    	}
    	if(flag==0) return digits;
    	}while(len>=0);
    	digits=new int[digits.length+1];
    	digits[0]=1;
    	return digits;
    }
//12月23号 20:57写
```

#### [387. 字符串中的第一个唯一字符](https://leetcode-cn.com/problems/first-unique-character-in-a-string/)

难度简单336

给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

**示例：**

```
s = "leetcode"
返回 0

s = "loveleetcode"
返回 2
```

对于这样的题目求持否重复字符，我们可以使用哈希表的思想来讲字母与数组进行映射，简单来说就是我们遍历数组然后统计每个字符的出现次数，在再次遍历的时候，我们取出第一个出现次数为1的位置就可以解决问题了，这道题是一个简单题，对于这样解决问题的办法的时间复杂度为O(N)(怎么感觉所有题目都可以被降到O（N）复杂度上) .

代码如下：

```java
    public int firstUniqChar(String s) {
    	int [] hashmap=new int[130];
    	for(int i=0;i<s.length();i++) {
    		hashmap[s.charAt(i)]++;
    	}
    	for(int i=0;i<s.length();i++) {
    		if(hashmap[s.charAt(i)]==1) return i;
    	}
    return -1;
    }
//12月23号20:58写
```

#### [135. 分发糖果](https://leetcode-cn.com/problems/candy/)

难度困难400

老师想给孩子们分发糖果，有 *N* 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

- 每个孩子至少分配到 1 个糖果。
- 相邻的孩子中，评分高的孩子必须获得更多的糖果。

那么这样下来，老师至少需要准备多少颗糖果呢？

**示例 1:**

```txt
输入: [1,0,2]
输出: 5
解释: 你可以分别给这三个孩子分发 2、1、2 颗糖果。
```

问题分析：对于这样的问题，我们分析一下，每个孩子自己必须分得到一颗糖果但是，相邻的孩子分数较高的会分到较多的糖果（太内卷了吧。。。），那么我们开始分析，对于这样的题目，我们可以先假设每个孩子都只会领到一个candy，然后我们再对这样的分配进行修改，那么我们分析规则，拆分为对于左到右开始的顺序，我们分析当自己左边的比自己小的时候我们要比左边的多一个糖果，但这样之后并不完美，我们还要从右到左再比较一次，审核所有结果，对于从右到左的顺序开始，当自己比右边孩子分数大的时候，我们查看自己是否糖果数比他多，来修改自己的糖果数，就这样左右扫描两次，时间复杂度O(N),n为孩子的个数，

```java
   public int candy(int[] ratings) {
    	int i=0,res=0;
    	int [] r=new int [ratings.length];
    	Arrays.fill(r, 1);
    	for(i=1;i<ratings.length;i++) {
    		if(ratings[i]>ratings[i-1]) {
    			r[i]=r[i-1]+1;
    		}	
    	}
    	for(i=ratings.length-2;i>=0;i--) {
    		if(ratings[i]>ratings[i+1]) {
    		r[i]=Math.max(r[i], r[i+1]+1);	
    		}
    	}
    	for(int k:r) {
    		res+=k;
    	}
    	return res;
    }
//12月24号 19时45分
```

#### [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

难度简单2059

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。

注意空字符串可被认为是有效字符串。

**示例 1:**

```
输入: "()"
输出: true
```

**示例 2:**

```
输入: "()[]{}"
输出: true
```

对于该题以及类似的匹配问题，我们可以一开始就想到用栈来求解，将一开始遍历的左括号用栈来压入，然后当开始匹配到右半边的时候，我们将当前元素出栈，来比较是否匹配，不匹配则返回错误，匹配则出栈，该代码时间复杂度为O(N),

```java
  public boolean isValid(String s) {
    	Stack<Character> st=new Stack<Character>();
    

    	for(char c:s.toCharArray()) {
    	  if(c=='(')st.push(')');
    		else if(c=='[') st.push(']');
    		else if(c=='{') st.push('}');
    		else if(st.isEmpty()||c!=st.pop()) return false;
    	
    	
    		
    	}
    	return st.isEmpty();
    
    }
//12月26号 11时57分
```

#### [100. 相同的树](https://leetcode-cn.com/problems/same-tree/)

难度简单

给定两个二叉树，编写一个函数来检验它们是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

**示例 1:**

```java
输入:       1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

输出: true
```

在对于树操作的时候，我们一般都是使用递归去探索树的左右节点，或者去操作他们，（不过说实话递归是真的费脑子不太好想这个解决问题的方法跟我们日常逻辑相反），那么我们对于求解这个问题，就比较左右节点的值是否相等，然后不断去往下递归就好了，代码很简单时间复杂度为O(N)(N的大小取决于树的节点数)。

```java
    public boolean isSameTree(TreeNode p, TreeNode q) {
    	if(p==null&&q==null) return true;
    	else if(p==null||q==null) return false;
    	else if(p.val!=q.val) return false;
    	return (isSameTree(p.left,q.left))&&(isSameTree(p.right,q.right));
    }
//12月26日 19时05分
```

#### [19. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

难度中等1156

给定一个链表，删除链表的倒数第 *n* 个节点，并且返回链表的头结点。

**示例：**

```java
给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.
```

​		对于这样的问题，好像有很简单的解法就是我们遍历链表求解整个链表的长度从而来知道我们到第几个要删除，但是这样的话我们需要两次遍历链表，那么如何只用一次遍历来解决这个问题。那么我们换种方法想想，如果我们让一个指针到达要删除的节点前一个节点是不是就可以了。

​	那么怎么完成了，这样的话我们需要两个指针利用倒数几个的性质来计算，使用双指针来确定删除节点的位置。我们唯一需要注意的就是倒数删除的节点为头结点，如果是这样的话，我们针对这个测试点，我们只需要返回头结点的下一个节点就可以了，这个代码的时间复杂度为O(n)代码如下：

```java
 public ListNode removeNthFromEnd(ListNode head, int n) {
            if(head==null||head.next==null) return null;
	    	ListNode lhead=head;

	    	ListNode ln=head;
	    	for(int i=0;i<n;i++) {
	    		ln=ln.next;		
	    	}
            if(ln==null) return head.next;
	    	while(ln.next!=null) {
	    		lhead=lhead.next;
	    		ln=ln.next;
	    	}
	    	lhead.next=lhead.next.next;
	    	return head;   }
```

#### [1046. 最后一块石头的重量](https://leetcode-cn.com/problems/last-stone-weight/)

难度简单123

有一堆石头，每块石头的重量都是正整数。

每一回合，从中选出两块 **最重的** 石头，然后将它们一起粉碎。假设石头的重量分别为 `x` 和 `y`，且 `x <= y`。那么粉碎的可能结果如下：

- 如果 `x == y`，那么两块石头都会被完全粉碎；
- 如果 `x != y`，那么重量为 `x` 的石头将会完全粉碎，而重量为 `y` 的石头新重量为 `y-x`。

最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 `0`。

**示例：**

```
输入：[2,7,4,1,8,1]
输出：1
解释：
先选出 7 和 8，得到 1，所以数组转换为 [2,4,1,1,1]，
再选出 2 和 4，得到 2，所以数组转换为 [2,1,1,1]，
接着是 2 和 1，得到 1，所以数组转换为 [1,1,1]，
最后选出 1 和 1，得到 0，最终数组转换为 [1]，这就是最后剩下那块石头的重量。
```

对于这样的问题，排序是不可避免的，但是可以边排序边缩小范围(这是一种想法)，但是我们在这里用另一种想法，用递归来表示，对于不同石头被粉碎的设定，我们可以这么假设，将被粉碎的石头一个用0来表示，另一个用他们之间做差来表示石头还未被粉碎的部分，原数组仍然保留，，然后对于还剩两块石头的时候，我们这么判断，当排序改后，倒数第三个数字为0的时候，说明这时只有两个石头没有被粉碎，我们直接返回他们的差，这样我们的程序就写完了，本程序使用了，递归加每次都排序了一次，时间复杂度在o(n^2logn)中，经过优化可以达到0-1ms的平均运行时间，具体代码如下：

```java
    public int lastStoneWeight(int[] stones) {
    	int len=stones.length;
    	if(len==1) return stones[0];
    	if(len==2) {
    		return Math.abs(stones[len-1]-stones[len-2]);
    	}
    	Arrays.sort(stones);
    	if(stones[len-3]==0) return (stones[len-1]-stones[len-2]);
    	stones[len-2]=stones[len-1]-stones[len-2];
    	stones[len-1]=0;
    	return lastStoneWeight(stones);     
    }
//12月30日17时18分
```

#### [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

难度中等329

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

**注意:**

1. 可以认为区间的终点总是大于它的起点。
2. 区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。

**示例 1:**

```
输入: [ [1,2], [2,3], [3,4], [1,3] ]

输出: 1

解释: 移除 [1,3] 后，剩下的区间没有重叠。
```

对于解决这样的问题，我们对于无序的操作总是非常烦心，所以为了让问题更加便于处理，我们需要首先对于问题进行排序，但是每个相应的输入都是二元组，那么我们需要怎么排序呢，我们可以这么想，如果按照第一个元素来进行排序，那么对于上一个元素，我们只需要比较前一个元素的结束范围是否大于下一个元素的起始范围，那么就说明他们之间发生了范围交叉，然后再删除之中，我们修改上界，为了能够删除最小的，我们选择他们之间下界较小的那个，如果没有交叉我们继续向下比较，这样时间复杂度由于用到了排序时间复杂度为O(nlogn)，具体代码如下：

```java
public int eraseOverlapIntervals(int[][] intervals) {
    	Arrays.sort(intervals,(a,b)->a[0]-b[0]);
    	int len=intervals.length;
        if(len==0) return 0;
    	int [] res=intervals[0];
    	int count=0;
    	for(int i=1;i<len;i++) {
    		if(intervals[i][0]<res[1]) {
    			count++;
    			res[1]=Math.min(intervals[i][1], res[1]);
    			
    			
    		}
    		else res=intervals[i];
    	}
    	return count;
  }
//1月1号13点十分
```

#### [605. 种花问题](https://leetcode-cn.com/problems/can-place-flowers/)

难度简单252

假设你有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花卉不能种植在相邻的地块上，它们会争夺水源，两者都会死去。

给定一个花坛（表示为一个数组包含0和1，其中0表示没种植花，1表示种植了花），和一个数 **n** 。能否在不打破种植规则的情况下种入 **n** 朵花？能则返回True，不能则返回False。

**示例 1:**

```java
输入: flowerbed = [1,0,0,0,1], n = 1
输出: True
```

对于这种要求附近不能拥有的我们可以看看对于一朵花他附近的结构一定是010，那栽种之前就是000，那么对于这样的情况，我们只需要统计是连续三个0的个数就好了，然后对于00000这样能种两枝花的我们将每次统计的总和count-2就可以了。

​	然后唯一需要注意的就是，对于开头和结尾这样的，我们默认他们自带一个0(他们的左侧，右侧不存在)如1,0,0,0,1，我们默认为0100010这样，就可以解决本次问题了，该问题的时间复杂度为O(N),平均运行时间为1ms

```java
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int count=1;
        
        for(int i=0;i<flowerbed.length;i++){
            if(flowerbed[i]==0) count++;
            else count=0;
            if(count==3){
                n--;
                count-=2;
            }     

        }
        if(count==2) n--;
        if(n<=0) return true;
        else return false;
    }
}
//1月1日 18时06分
```

#### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

难度困难754

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

 **示例 1：**

```
输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

​	对于该问题我想到有两种解法，一直就是最暴力的方法，就是按照题目所说的，我们每次提取滑动窗口中的所有数字，然后进行排序之后选取最大的，如此往复这样的时间复杂度为O(n*(k+klogk))，具体代码如下：

```java
 public int[] maxSlidingWindow(int[] nums, int k) {
    	int [] res=new int[nums.length-k+1];
    	LinkedList<Integer> list=new LinkedList();
    	int i;
    	for( i=0;i<=nums.length-k;i++) {
    		System.arraycopy(nums, 0, tmp, 0, k);
    		Arrays.sort(tmp);
    		res[i]=tmp[k-1];
    			
    	}
    	return res;

    }
```

但是这样的话时间复杂度过高。。。，题目的限制应该是要在O(n)的情况下，解决该题目，那么介绍对于该题目的另一种方法，利用java的双端队列LinkedList

​	在这个题目中，我们维护了一个递减的双端队列（即它的头部为每个滑动区间的最大值），为了保证他是递减的，我们每次新添加元素的时候，比较该元素加入与前面的元素大小，比他小的全部删除，之后再插入该元素，当滑动窗口的最左端超出窗口时，我们将他删除，当窗口达到K时，我们取队列的头部，这样的话时间复杂度在O(n)中，代码如下:

```java
 public int[] maxSlidingWindow(int[] nums, int k) {
    	int [] res=new int[nums.length-k+1];
    	LinkedList<Integer> list=new LinkedList();
    	int i;
    	for( i=0;i<nums.length;i++) {
    		while(!list.isEmpty()&&nums[list.peekLast()]<=nums[i]) {
    			list.pollLast();
    		}
    		list.addLast(i);
    		if(list.peekFirst()<=i-k) list.pollFirst();
    		if(i-k+1>=0)
    			res[i-k+1]=nums[list.peekFirst()];
    		
    		
    	}
    	return res;

    }
//2021年1月2日 22时21分
```

#### [86. 分隔链表](https://leetcode-cn.com/problems/partition-list/)

难度中等

​	给你一个链表和一个特定值 `x` ，请你对链表进行分隔，使得所有小于 `x` 的节点都出现在大于或等于 `x` 的节点之前。

你应当保留两个分区中每个节点的初始相对位置。

 **示例：**

```
输入：head = 1->4->3->2->5->2, x = 3
输出：1->2->2->4->3->5
```

​	对于这个题目，其实审题就可以看出来具体怎么做了，要求保留每个节点的初始相对位置，那么他的意思是使用排序来做一定会报错，也就是要求我们在不改变原来顺序的情况下，一次遍历就可以完成，那么我们就可以有个想法用两个指针，一个连接比指定值小的节点，一个连接比指定值大的节点来最后连接在一起，题目不就完成了？（很难想象这道题是中等。。。），时间复杂度一次遍历O(n),具体代码如下：

```java
    public ListNode partition(ListNode head, int x) {
		ListNode small=new ListNode();
		ListNode big=new ListNode();
		ListNode h1=small;
		ListNode h2=big;
		while(head!=null){
			if(head.val<x) {
                small.next=new ListNode(head.val);
				small=small.next;
		}
		else {
			
			big.next=new ListNode(head.val);
			big=big.next;
		}
		head=head.next;
	}
	small.next=h2.next;
	return h1.next;
}
```

#### [189. 旋转数组](https://leetcode-cn.com/problems/rotate-array/)

给定一个数组，将数组中的元素向右移动 *k* 个位置，其中 *k* 是非负数。

**示例 1:**

```
输入: [1,2,3,4,5,6,7] 和 k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]
```

对于这道题，其实我们第一种解法是一种非常简单的思想，竟然是右移，那么我们可以根据右移发现一个规律，即移动后第i个元素的位置，在(i+k)%nums.length上，那么我们只需要额外创建以个数组然后对这个数组按照这个规则进行赋值即可，最后在将数值返回赋值，这样的时间复杂度为O(N)，平均运行时间1ms：具体代码：

```java
 public void rotate(int[] nums, int k) {
    int	len =nums.length;
    int [] res=new int[len];
    int index;
    for(int i=0;i<len;i++) {
    	index=(i+k)%len;
    	res[index]=nums[i];
    }
    for(int i=0;i<len;i++){
        nums[i]=res[i];
    }}
//2021年1月8日22时57分
```

第二种方法是我们可以将数组的右移看成一次数组前后对转（我想这也是题目标题的原意），也就是对于1234567,k=3, 我们对于他的运算，可以看做是先前后对转为7654321，接下来k=3将数组划分区域为765 ,4321我们在对他们进行旋转即可得到567 1234(之所以还要在旋转是为了保持原有顺序保持不变，这个时间复杂度为O(n)，但它相对于上面那个算法是减少了使用次数，平均运行时间0ms，具体代码如下：

```java
 public void rotate(int[] nums, int k) {
		int len=nums.length;
		reverse(nums, 0,len-1);
		reverse(nums,0,k-1);
		reverse(nums, k,len-1);

	}
	
	public void reverse(int [] nums,int l,int r){
		int tmp;
		while(l<r){
		tmp=nums[l];
		nums[l++]=nums[r];
		nums[r--]=tmp;			
		}
	}
//2021年1月8日 22时57分
```

#### [123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

难度困难605

给定一个数组，它的第 `i` 个元素是一支给定的股票在第 `i` 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 **两笔** 交易。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

 

**示例 1:**

```
输入：prices = [3,3,5,0,0,3,1,4]
输出：6
解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
```

对于这个版本，我依然喜闻乐见地采用动态规划方程，来解决该问题，我们使用的的方程采用相当简单的逻辑，我们就直接上代码了：

```java
  public int maxProfit(int[] prices) {
		int len=prices.length-1;
		int [] result=new int[2];
		result[0]=profit(prices,len,4,0);
		return result[0];

	}
	public int profit(int []prices,int n,int k,int flag){
		if(k==0) return 0;
			if(n==0&&flag==0) return 0;
			if(n==0&&flag==1) return -prices[0];
		if(flag==1){  
	return Math.max(profit(prices,n-1,k-1,0)-prices[n],profit(prices,n-1,k,1));
		
		}
		else {
	return Math.max(profit(prices,n-1,k-1,1)+prices[n],profit(prices,n-1,k,0));
		}		
	}
//2021年1月九日11时20分
```

但这样的是在通不过测试（应该是达到了2的n次方？），那么，我们只好换种方法，考虑他四个状态(因为题目交代最多交易两次)，那么就有五个状态，因为第一次买了还没卖，第一次买了然后卖，第一次卖了然后买，第二次买然后卖，我们求解出这样的方程然后求解第二次买然后卖即可(因为这样第二次的状态来自第一次的状态)：所以我们写出一下的代码：

```java
     public int maxProfit(int[] prices) {
		int buy1,buy2,sell1,sell2;
		buy1=-prices[0];
		buy2=-prices[0];
		sell1=0;
		sell2=0;
		for(int i=1;i<prices.length;i++){
			buy1=Math.max(buy1,-prices[i]);
			sell1=Math.max(sell1,buy1+prices[i]);
			buy2=Math.max(buy2,sell1-prices[i]);
			sell2=Math.max(sell2,buy2+prices[i]);
		}
		return sell2;
	}
//2021年1月9日11时19分
```

#### [228. 汇总区间](https://leetcode-cn.com/problems/summary-ranges/)

难度简单100

给定一个无重复元素的有序整数数组 `nums` 。

返回 **恰好覆盖数组中所有数字** 的 **最小有序** 区间范围列表。也就是说，`nums` 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 `nums` 的数字 `x` 。

列表中的每个区间范围 `[a,b]` 应该按如下格式输出：

- `"a->b"` ，如果 `a != b`
- `"a"` ，如果 `a == b`

对于这个题目，可以利用双指针来进行解决，两个指针low,high，每次比较后面的是否比前面的大一，然后来移动，题目思想也较为简单，时间复杂度O(n),平均运行时间为0ms，

```java
public List<String> summaryRanges(int[] nums) {
		List<String> result = new ArrayList<String>();
		int low,high,i=0;
		while(i<nums.length){
			low=i;
			i++;
			while(i<nums.length&&nums[i]==nums[i-1]+1){
				i++;
			}
			high=i-1;
			StringBuffer s=new StringBuffer();
			s.append(Integer.toString(nums[low]));
			if(low<high){
				s.append("->");
				s.append(Integer.toString(nums[high]));
			}
			result.add(s.toString());
			
		}
		return result;
	}
//2020年1月10号 11时11分
```

#### [783. 二叉搜索树节点最小距离](https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/)

难度简单161

给你一个二叉搜索树的根节点 `root` ，返回 **树中任意两不同节点值之间的最小差值** 。

**注意：**本题与 530：https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/ 相同

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/02/05/bst1.jpg)

```
输入：root = [4,2,6,1,3]
输出：1
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/02/05/bst2.jpg)

```
输入：root = [1,0,48,null,null,12,49]
输出：1
```

 

**提示：**

- 树中节点数目在范围 `[2, 100]` 内

- `0 <= Node.val <= 105`

  分析：

对于这个常见的树的问题，应该想到是用一个深度遍历算法DFS来解决该问题，因为对于树这种数据结构的问题，如何遍历他的左节点和右节点，总是我们烦恼的问题，那么对于这样的想法，我们可以使用树的中序遍历，即先遍历左子树，操作放在访问左子树后右字数前为中序遍历，所以我们如此编写代码：

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int pre,ans;
    
    public int minDiffInBST(TreeNode root) {
        ans=Integer.MAX_VALUE;
        pre=-1;
        dfs(root);
        return ans;
    }
    public void dfs(TreeNode root){
        if(root==null) return;
        dfs(root.left);
        if(pre==-1){
            pre=root.val;
        }
        else{
            ans=Math.min(ans,root.val-pre);
            pre=root.val;
        }
        dfs(root.right);
    }
}
```

#### [1310. 子数组异或查询](https://leetcode-cn.com/problems/xor-queries-of-a-subarray/)

难度中等108

有一个正整数数组 `arr`，现给你一个对应的查询数组 `queries`，其中 `queries[i] = [Li, Ri]`。

对于每个查询 `i`，请你计算从 `Li` 到 `Ri` 的 **XOR** 值（即 `arr[Li] **xor** arr[Li+1] **xor** ... **xor** arr[Ri]`）作为本次查询的结果。

并返回一个包含给定查询 `queries` 所有结果的数组。

 

**示例 1：**

```
输入：arr = [1,3,4,8], queries = [[0,1],[1,2],[0,3],[3,3]]
输出：[2,7,14,8] 
解释：
数组中元素的二进制表示形式是：
1 = 0001 
3 = 0011 
4 = 0100 
8 = 1000 
查询的 XOR 值为：
[0,1] = 1 xor 3 = 2 
[1,2] = 3 xor 4 = 7 
[0,3] = 1 xor 3 xor 4 xor 8 = 14 
[3,3] = 8
```

**示例 2：**

```tex
输入：arr = [4,8,2,10], queries = [[2,3],[1,3],[0,0],[0,3]]
输出：[8,0,4,4]
```

解析：对于这道题，使用位运算的异或操作，对于这种题目，乍一看好像挺简单的，好像我们按照题目的方法直接异或操作就可以了

```python
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        result=[]
        for query in queries:
            l=query[0]
            r=query[1]
            res=0
            while l<=r:
                res=res^arr[l]
                l=l+1
            result.append(res)
        return result
```

但题目当然没有这么简单，题目对于时间上的限制，使得我们并不能使用这种复杂的方法

我们的解决方向可以利用位运算的对称性来解决，即a^b^c^a^b=c,所以我们的想法就是努力去构造这样的方法，如题如果arr=[a,b,c,d]，我们想要求[1,3]即啊b^c^d，那么假设我们获得了一个数组，res=[0,a,a^b,a^b^c,a^b^c^d],那么我们要求得结果是不是只需要求res[1]^res[4]=a^a^b^c^d=b^c^d是不是就可以了，而构造这种累加的数组对于使用python迭代器的accumlate是非常简单的所以对于这道题只需要两行代码就可以完成了

```python
from itertools import accumulate
from operator import xor
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        ans=list(accumulate([0]+arr,xor))
        return [ans[l]^ans[r+1] for l,r in queries]#这是一个列表推导式
```

#### [12. 整数转罗马数字](https://leetcode-cn.com/problems/integer-to-roman/)

难度中等601

罗马数字包含以下七种字符： `I`， `V`， `X`， `L`，`C`，`D` 和 `M`。

```
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

例如， 罗马数字 2 写做 `II` ，即为两个并列的 1。12 写做 `XII` ，即为 `X` + `II` 。 27 写做 `XXVII`, 即为 `XX` + `V` + `II` 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 `IIII`，而是 `IV`。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 `IX`。这个特殊的规则只适用于以下六种情况：

- `I` 可以放在 `V` (5) 和 `X` (10) 的左边，来表示 4 和 9。
- `X` 可以放在 `L` (50) 和 `C` (100) 的左边，来表示 40 和 90。 
- `C` 可以放在 `D` (500) 和 `M` (1000) 的左边，来表示 400 和 900。

给你一个整数，将其转为罗马数字。

 

**示例 1:**

```
输入: num = 3
输出: "III"
```

**示例 2:**

```
输入: num = 4
输出: "IV"
```

**示例 3:**

```
输入: num = 9
输出: "IX"
```

**示例 4:**

```
输入: num = 58
输出: "LVIII"
解释: L = 50, V = 5, III = 3.
```

**示例 5:**

```
输入: num = 1994
输出: "MCMXCIV"
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

对于这道题其实可以看出来，罗马数字是一种累加的数字，区别于我们的一般十进制，比如21=2*10+1而罗马数字21表示则是'XXI'因为X+X+I=10+10+1=21，所以我们可以相处这种办法就是，比如我们从第一个小于21的数开始加如：21-10=11，res='X',此时11>10,所以我们仍然用十去减11-10=1,res='XX',这时1<10 1>=1，1-1=0，res='XXI',这就是我们的结果，然后这题最需要注意的就是，有些特殊数字如果纠结于何时出现这种状况，我们的算法将是非常复杂的，所以我们可以把他们也看作一种特殊的数字从而像其他的罗马数字一样去代替：

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        values=[1000,900,500,400,100,90,50,40,10,9,5,4,1]
        romaStr=['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
        res=''
        for i in range(len(romaStr)):
            while num>=values[i]:
                num=num-values[i]
                res=res+romaStr[i]
        return res


#写于2021 5月14日 23时09分
```

#### [13. 罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/)

难度简单1321

罗马数字包含以下七种字符: `I`， `V`， `X`， `L`，`C`，`D` 和 `M`。

```
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

例如， 罗马数字 2 写做 `II` ，即为两个并列的 1。12 写做 `XII` ，即为 `X` + `II` 。 27 写做 `XXVII`, 即为 `XX` + `V` + `II` 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 `IIII`，而是 `IV`。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 `IX`。这个特殊的规则只适用于以下六种情况：

- `I` 可以放在 `V` (5) 和 `X` (10) 的左边，来表示 4 和 9。
- `X` 可以放在 `L` (50) 和 `C` (100) 的左边，来表示 40 和 90。 
- `C` 可以放在 `D` (500) 和 `M` (1000) 的左边，来表示 400 和 900。

给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

 

**示例 1:**

```
输入: "III"
输出: 3
```

**示例 2:**

```
输入: "IV"
输出: 4
```

**示例 3:**

```
输入: "IX"
输出: 9
```

**示例 4:**

```
输入: "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.
```

**示例 5:**

```python
输入: "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

这题初看是和上题目一相反的，以为是将上题相反的方向来反转从而解决问题，但是这里的我们需要注意一件事情，就是，对于这道题分情况讨论并不是最好的方法，我们不如仔细看下问题，我们需要注意的是IV,CM这些特殊点，仔细观察可以发现，这些的共同点就是左边字母代表的数字，比右边代表的字母要小与其他并不一样，所以我们的代码可以向如下编写：

```python
class Solution:
    def romanToInt(self, s):
        a = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}        
        res=0        
        for i in range(len(s)):            
            if i<len(s)-1 and a[s[i]]<a[s[i+1]]:                
                res-=a[s[i]]
            else:
                res+=a[s[i]]
        return res
    #写于2021年5月15日11时30分
```

#### [993. 二叉树的堂兄弟节点](https://leetcode-cn.com/problems/cousins-in-binary-tree/)

难度简单194

在二叉树中，根节点位于深度 `0` 处，每个深度为 `k` 的节点的子节点位于深度 `k+1` 处。

如果二叉树的两个节点深度相同，但 **父节点不同** ，则它们是一对*堂兄弟节点*。

我们给出了具有唯一值的二叉树的根节点 `root` ，以及树中两个不同节点的值 `x` 和 `y` 。

只有与值 `x` 和 `y` 对应的节点是堂兄弟节点时，才返回 `true` 。否则，返回 `false`。

 

**示例 1：
![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/16/q1248-01.png)**

```
输入：root = [1,2,3,4], x = 4, y = 3
输出：false
```

**示例 2：
![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/16/q1248-02.png)**

```
输入：root = [1,2,3,null,4,null,5], x = 5, y = 4
输出：true
```

**示例 3：**

**![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/16/q1248-03.png)**

```
输入：root = [1,2,3,null,4], x = 2, y = 3
输出：false
```

 

**提示：**

- 二叉树的节点数介于 `2` 到 `100` 之间。

- 每个节点的值都是唯一的、范围为 `1` 到 `100` 的整数。

  对于这道题，我们的解法可以分析一下寻找是否是堂兄弟的关键点是这两个节点的深度相同，并且他们的父节点不一样，我们对于二叉树仍然是用深度优先遍历为主，那么我们就要思考在遍历的时候，需要在多带两个参数，深度和父节点，所以我们的代码可以这么编写

  ```python
  class TreeNode:
      def __init__(self, val=0, left=None, right=None):
          self.val = val
          self.left = left
          self.right = right
  class Solution:
      def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
      	x_parent,x_found,x_depth=None,False,None
      	y_parent,y_found,y_depth=None,False,None
      	def dfs(root,depth,parent):
      		if not root:
      			return
      		nonlocal x_parent,x_found,x_depth,y_parent,y_depth,y_found#声明这些变量并不是局部变量
      		if root.val==x:
      			x_parent,x_found,x_depth=parent,True,depth
      		elif root.val==y:
      			y_parent,y_found,y_depth=parent,True,depth
      		if x_found and y_found:#当找到两个节点的时候，我们返回
      			return
      		dfs(root.left,depth+1,root)
      		if x_found and y_found:
      			return
      		dfs(root.right,depth+1,root)
      	dfs(root,0,None)
      	return x_depth==y_depth and x_parent!=y_parent#比较深度相同，且父节点不相同
  #写于2021年五月十七日
  ```

  