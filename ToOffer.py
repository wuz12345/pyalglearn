"""
剑指Offer的题目
"""
def findRepeatNumber(nums) -> int:
    """
    第一题，数组重复问题
    :param nums: List
    :return:
    """
    from collections import defaultdict
    ans = defaultdict(int)
    for i in nums:
        ans[i] = ans[i]+1
        if ans[i]>1:
            return i
    return -100
def findRepeatNumber_1(nums) -> int:
    """
    第一题，数组重复问题
    :param nums: List
    :return:
    """
    l = []
    for i in nums:
        if i not in l:
            l.append(i)
        else:
            return i
    return -100
def findNumberIn2DArray(matrix, target) -> bool:
    import numpy as np
    temp = np.array(matrix)
    row,col = temp.shape
    if row == 1 and col ==1:
        return temp[0][0] == target
    if target < temp[row//2][col//2]:
        return findNumberIn2DArray(list(temp[0:row//2][0:col//2]),target)


#除法求值
def calcEquation( equations, values, queries):
    from collections import defaultdict
    graph = defaultdict(int)
    vex = []
    for i in range(len(values)):
        a = equations[i][0]
        b = equations[i][1]
        graph[(a,b)] = values[i]
        graph[(b,a)] = 1/values[i]
        vex.append(a)
        vex.append(b)

    vexset = set(vex)
    for i in vexset:
        for j in vexset:
            for k in vexset:
                if graph[(j,i)] and graph[(i,k)]:
                    graph[(j,k)] = graph[(j,i)]  *graph[(i,k)]

    ans = []
    for i in queries:
        if graph[(i[0],i[1])]:
            ans.append(graph[(i[0],i[1])])
        else:
            ans.append(-1)
    return ans



