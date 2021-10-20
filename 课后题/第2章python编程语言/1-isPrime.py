# 2.1 编写 isprime()函数,参数为整数,并且需要有异常处理功能。
# 此函数的功能是检测接收的整数是否为质数, 如果整数是质数, 则返回True, 否则返回 False。

#!/usr/bin/env python
# -*- coding:utf-8 -*-


def isPrime(variate):
    # 质数大于 1，2，3，
    if num in [1, 2, 3]:
        return True

    if num > 3:
        for i in range(2, int(num / 2)):
            if (num % i) == 0:
                return False
                break
        else:
            return True

    # 如果输入的数字小于或等于 1，不是质数
    else:
        return False


# 当你想在房间里找某样东西时 ， 只要在任意位置找到了， 就停止继续搜查工作 。
# 但如果把整个房间都翻遍了， 还没找到我们想要的东西时， 需要告诉人家， 很抱歉，
# 这儿没有你要找的东西 。 遇到这样的情况时就用 for ... else ，

# 除此之外 ， 恐怕只会引起误操作 。

# Python之父为什么要搞出这样的一种语法糖出来呢 ？ 这是我们常人没法理解的 。
# 不过「python之禅」告诉了我们答案： "Although that way may not be obvious at first unless you're Dutch."。


# def isPrime():
#     sites = ["Baidu", "Google", "Taobao"]
#     for site in sites:
#         # if site == "Runoob":
#         #     print("菜鸟教程!")
#         #     break
#         print("循环数据 " + site)
#     else:
#         print("没有循环数据!")
#     print("完成循环!")

if __name__ == '__main__':
    try:
        num = int(input("请输入一个数字: "))
        print(isPrime(num))
    except ValueError:
        print("您输入的不是数字，请再次尝试输入！")
