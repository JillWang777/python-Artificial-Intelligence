
# 4.编写程序实现随机密码生成 。
# 要求在26个大小写字母和10个数字组成的列表中随机生成10个8位密码。

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

# random.sample #函数的用法？？？？？
