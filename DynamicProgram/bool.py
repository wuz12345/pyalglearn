
def compare_string(string1: str, string2: str) -> str:
    """
    >>> compare_string('0010','0110')
    '0_10'

    >>> compare_string('0110','1101')
    -1
    如示例所见，函数能找出字符串中相同的部分，并将不同的部分用下划线代替；
    若函数每个位置都不相同，则返回-1
    """

    l1 = list(string1)
    l2 = list(string2)
    assert len(l1) == len(l2) ,"The Length Of Two String Must Equal!"
    count = 0
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            count += 1



            l1[i] = "_"
    if count > 1:
        return -1
    else:
        return "".join(l1)



if __name__ =="__main__":
    print(compare_string('1110','111'))