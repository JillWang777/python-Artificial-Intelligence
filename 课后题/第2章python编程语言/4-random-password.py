
# 4.编写程序实现随机密码生成 。
# 要求在26个大小写字母和10个数字组成的列表中随机生成10个8位密码。

# [chr(x) for x in range(ord('A'), ord('Z')+1)]
# 列表的取值形式

# random.sample #函数的用法？？？？？
# 返回从总体序列或集合中选择的唯一元素的 k 长度列表。
# 用于无重复的随机抽样。


# 可以总结一下Python函数对变量的作用要遵守的原则：
# （1）简单数据类型变量无论是否与全局变量重名，仅在函数内部创 建和使用。函数退出后，变量就会被释放，而同名的全局变量不受函数调
# 用影响。
# （2）简单数据类型变量在使用global保留字声明后，作为全局变量
# 使用，函数退出后，该变量仍被保留，且数值被函数改变。
# （3）对于组合数据类型的全局变量，如果在函数内部没有被真实地
# 创建同名变量，则函数内部可以直接使用并修改全局变量的值。
# （4）如果函数内部真实地创建了组合数据类型变量，无论是否与全 局变量同名，函数仅对内部的局部变量进行操作，函数退出后局部变量被
# 释放 ， 而全局变量的值不受函数影响 。


import random
passelect = []
for j in [chr(x) for x in range(ord('a'), ord('z')+1)]:
    passelect.append(j)
for j in [chr(x) for x in range(ord('A'), ord('Z')+1)]:
    passelect.append(j)
for j in [x for x in range(0, 10)]:  # for i in range(10)也行
    passelect.append(str(j))
for i in range(10):
    list1 = random.sample(passelect, 8)  # 从passelect列表中随机选8个
    for p in list1:
        print(p, end='')
    print()

[x for x in [1, 2, 3, 4]]
