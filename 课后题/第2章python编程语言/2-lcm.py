# 最小公倍数

# 定义函数
def lcm(x, y):

    #  获取最大的数
    if x > y:
        greater = x
    else:
        greater = y

    while(True):
        if((greater % x == 0) and (greater % y == 0)):
            lcm = greater
            break
        greater += 1

    return lcm


# 用户输入两个数字
try:
    num1 = int(input("输入第一个数字: "))
    num2 = int(input("输入第二个数字: "))
    print(num1, "和", num2, "的最小公倍数为", lcm(num1, num2))
except ValueError:
    print("您输入的不是数字，请再次尝试输入！")
