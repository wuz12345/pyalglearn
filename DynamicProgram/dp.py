# 这个文件主要关于动态规划问题
# 问题如下：
# 1、硬币兑换问题
def coin_change(S,n):
    """
    硬币兑换可以看作是找零问题：
    假设你是一家超市的售货员，你需要为顾客找零N元，而你只有固定面额但不限数量的纸币
    请问共有几种方式找零？
    例如：你需要找零4元，不限量的固定纸币面额为1元、2元、3元，则共有以下几种方法：
    [1,3],[2,2],[2,1,1],[1,1,1,1]
    再比如，需要找零十元，纸币面额为2、3、5，则有以下几种方法:
    [2,3,5],[2,2,2,2,2],[5,5],[3,3,2,2]
    :param S: 纸币面额
    :param n: 找零面额
    :return: 次数

    """
    if n < 0:
        return -1
    table = [0]*(n+1)
    """
    解决思路：
    本题是一个动态规划问题
    """
    table[0] = 1
    for value in S:
        for j in range(value,n+1):
            table[j] += table[j-value]

    return table[n]
def abbr(a: str, b: str) -> bool:
    """
    """
    n = len(a)
    m = len(b)
    dp = [[False for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = True
    for i in range(n):
        for j in range(m + 1):
            if dp[i][j]:
                if j < m and a[i].upper() == b[j]:
                    dp[i + 1][j + 1] = True
                if a[i].islower():
                    dp[i + 1][j] = True
    return dp[n][m]
if __name__=="__main__":
    print(abbr("acdbng","CNG"))


