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
import networkx as nx

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
字符串相乘，使用字符串相加算法
"""
# def multiply( num1: str, num2: str) -> str:

"""
较大分组的位置
"""
# def largeGroupPositions( s: str):
#     ret = list()
#     n, num = len(s), 1
#
#     for i in range(n):
#         if i == n - 1 or s[i] != s[i + 1]:
#             if num >= 3:
#                 ret.append([i - num + 1, i])
#             num = 1
#         else:
#             num += 1
#
#     return ret
#

#较大分组
def largeGroupPositions(s: str):
    nl = 1
    ans = []
    for i in range(len(s)):
        if i==len(s)-1 or s[i]!=s[i+1]:
            if nl>=3:
                ans.append([i-nl+1,i])
            nl=1
        else:
            nl+=1
    return ans
#Z字形变换
def convert( s: str, numRows: int):
    if numRows == 1:
        return s
    sl = len(s)
    ans = []
    index = 1
    dir = True
    xxx={}
    for i in range(numRows):
        xxx[i+1] = []
    for i in range(sl):
        if dir == True:
            ans.append(index)
            index+=1
            if index==numRows:
                dir = False
        if dir == False :
            ans.append(index)
            index -=1
            if index == 1:
                dir = True
    for i in range(sl):
        xxx[ans[i]].append(s[i])
    xxxx = ""
    for i in range(numRows):
        xxxx = xxxx + ''.join(xxx[i+1])
    print(xxxx)
    print(ans)

#是有效数独

# def isValidSudoku(self, board):
#     row = [{}for i in range(9)]
#     col = [{}for i in range(9)]
#     box = [{}for i in range(9)]
#     for i in range(9):
#         for j in range(9):
#             if board[i][j] != '.':
#                 num = int(board[i][j])
#                 box_index = (i//3)*3 + j//3
#                 rows[i][num] = rows[i].get(num, 0) + 1
#                 columns[j][num] = columns[j].get(num, 0) + 1
#                 boxes[box_index][num] = boxes[box_index].get(num, 0) + 1
#
#                 if rows[i][num] > 1 or columns[j][num] > 1 or boxes[box_index][num] > 1:
#                     return False
#     return True

#数字条约
def canJump(nums) -> bool:
    max_pos = nums[0]+0
    for i in range(1,len(nums)):
        if i <= max_pos:
            if i+nums[i] >= max_pos:
                max_pos = i+nums[i]

    return max_pos>=len(nums)-1

#能否环路
#起始点的汽油两要大于起始点的消耗
def canCompleteCircuit(gas, cost):
    for i in range(len(gas)):
        if gas[i]>=cost[i]:
            if isCircuit(i,gas,cost):
                return i
    return -1
def isCircuit(i,gas,cost):
    endpos = -1
    currentpos = i
    currentoil = gas[i]
    totalnums = len(gas)
    for j in range(totalnums):
        if currentoil >=cost[currentpos]:
            currentpos = currentpos+1 if(currentpos+1 <totalnums) else 0
            lastpos = currentpos-1 if(currentpos-1>=0) else totalnums-1
            currentoil = currentoil - cost[lastpos] + gas[currentpos]
            endpos = currentpos

    return endpos
#链表问题
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        return ListNode(0)
#有效字符串

#最大水容量

def maxArea(height) -> int:
    nums = len(height)
    start = 0
    end = nums-1
    maxa = -9999
    while start <= end:
        if maxa < Aera(start,height[start],end,height[end]):
            maxa = Aera(start,height[start],end,height[end])
        if height[start] > height[end]:
            end -= 1
        else:
            start+=1
    return maxa

def Aera(x1,y1,x2,y2):
    return (x2-x1)*min(y1,y2)
#外观数列
def countAndSay(n: int) -> str:
    if n==1:
        return "1"
    else:
        tempstr = countAndSay(n-1)
        return cutSlpit(tempstr)
def cutSlpit(s):
    nums = len(s)
    ans=[]
    temp = ""
    for i in range(nums):
        temp+=s[i]
        if i<= nums-2 and s[i] == s[i+1]:
            continue
        ans.append(temp)
        temp = ""
    anstr = ""
    for i in ans:
        anstr = anstr+str(len(i))+str(i[0])
    return anstr

#省份问题
def findCircleNum(isConnected) -> int:
    import numpy as np
    isConnectedMatrix = np.matrix(isConnected)
    row,col = isConnectedMatrix.shape
    DegreeMatrix = np.identity(row)
    for i in range(row):
        DegreeMatrix[i][i] = np.sum(isConnectedMatrix[i])
    LaplaceMatrix = DegreeMatrix - isConnectedMatrix
    x,_ = np.linalg.eigh(LaplaceMatrix)
    return list(x).count(0)
#字符串替换问题
def replaceSpace(s: str) -> str:
    sl = list(s)
    for i in range(len(sl)):
        if sl[i] == " ":
            sl[i] = "%20"
    return "".join(sl)

#前n个高频数字
def topKFrequent(nums) :
    from collections import defaultdict
    ans = defaultdict(int)
    for i in nums:
        ans[i] = ans[i]+1
    #用于保存前k个值
    print(ans)
#旋转数组
def rotate(nums,k) :
    length = len(nums)
    ans = [0 for i in range(len(nums))]
    for i in range(len(nums)):
        ans[i] = nums[(i+k)%length]
    return ans
#中位数
def findMedianSortedArrays(nums1, nums2) :
    len1 = len(nums1)
    len2 = len(nums2)
    total = [0 for i in range(len1 + len2)]
    index = 0
    index1 = 0
    index2 = 0
    while index1 < len1 and index2 < len2:
        if nums1[index1] < nums2[index2]:
            total[index] = nums1[index1]
            index1 += 1
        else:
            total[index] = nums2[index2]
            index2 += 1
        index += 1
    total[index:] = nums1[index1:] if index < len1 else nums2[index2:]
    print(total)
    a = (len1 + len2) % 2
    if a == 0:
        b = (total[(len1 + len2) // 2] + total[(len1 + len2) // 2 - 1])
        print(b/ 2)
    else:
        print(total[(len1 + len2) // 2])
#最大子序数和
def maxSubArray( nums) -> int:
    f = [0 for i in range(len(nums))]
    currentmax = f[0]
    f[0] = nums[0]
    for i in range(1,len(nums)):
        f[i] = max(f[i-1]+nums[i],nums[i])
        if currentmax < f[i]:
            currentmax = f[i]
    print(currentmax)
#华东窗口最大值
def maxSlidingWindow( nums, k):
    import heapq
    n = len(nums)
    # 注意 Python 默认的优先队列是小根堆
    q = [(-nums[i], i) for i in range(k)]
    heapq.heapify(q)

    ans = [-q[0][0]]
    for i in range(k, n):
        heapq.heappush(q, (-nums[i], i))
        while q[0][1] <= i - k:
            heapq.heappop(q)
        ans.append(-q[0][0])

    print(ans)

#全排列
def permute(nums):
    import itertools
    ans = []
    for num in itertools.permutations(nums):
        print(list(num))
#最大股票收益
def maxProfit(prices):
    nums = len(prices)
    x = []
    if nums == 1:
        return  0
    for i in range(1,len(prices)):

        if prices[i]>prices[i-1]:
            x.append(1)
        if prices[i]<prices[i-1]:
            x.append(-1)
        if prices[i] == prices[i-1]:
            x.append(0)
    if sum(x) == nums-1:
        return prices[-1] - prices[0]
    if sum(x) == 1-nums:
        return 0
if __name__ == "__main__":
    print(maxProfit( [7,6,4,3,1] ))