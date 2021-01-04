"""
本文件包含所有字符串算法
"""

# 给定两个字符串形式的非负整数num1 和num2，计算它们的和。
#
#
#
# 提示：
#
# num1 和num2的长度都小于 5100
# num1 和num2 都只包含数字  0-9
# num1 和num2 都不包含任何前导零
# 你不能使用任何內建 BigInteger 库，也不能直接将输入的字符串转换为整数形式

def Reverse(s:str) -> str:
    """
    字符串逆
    :param s:
    :return:
    """
    num = len(s)
    ans = ""
    for i in range(num-1,-1,-1):
        ans = ans + s[i]
    return ans
def padding(s1:str,s2:str):
    len1 = len(s1)
    len2 = len(s2)
    if len2==len1:
        return s1,s2
    if len2<len1:
        for i in range(len2,len1):
            s2+="0"
        return s1,s2
    if len2 > len1:
        for i in range(len1, len2):
            s1 += "0"
        return s1, s2

def addStrings( num1: str, num2: str) -> str:
    """
    :param num1: 字符串1
    :param num2: 字符串2
    :return:
    """
    num1_re = Reverse(num1)
    num2_re = Reverse(num2)
    num1_re,num2_re = padding(num1_re,num2_re)
    len1 = len(num1_re)
    len2 = len(num2_re)
    ans = ""
    index1 = 0
    addt= 0
    while True:
        add = int(num1_re[index1])+int(num2_re[index1]) + addt
        addt = 1 if add >=10 else 0
        adds = str(add) if add<10 else str(add%10)
        ans = ans + adds
        index1+=1
        if index1>=len1 and addt == 0:
            return Reverse(ans)
        if index1>=len1 and addt == 1:
            return Reverse(ans+"1")

def addstr( num1: str, num2: str):
    i,j = len(num1)-1,len(num2)-1
    ans = []
    add = 0
    while i>=0 or j>=0 or add!=0:
        x = int(num1[i]) if i>=0 else 0
        y = int(num2[j]) if j>=0 else 0
        result = x+y+add
        add = result // 10
        result = str(result) if result<10 else str(result%10)
        ans.append(result)
        i-=1
        j-=1
    return "".join(ans[::-1])

"""
字符串相乘
"""







print(addstr("123","1239"))