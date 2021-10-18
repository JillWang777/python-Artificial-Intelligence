# 6、用Newton法求解无约束优化问题的Python代码实现


def fd(x):
    t = np.asarray([2, 4])
    #y = np.dot(x.T,t)
    y = x.T * t
    return y


def fdd():
    #ys = 12*x**2-24*x-12
    a = np.asarray([[2, 0], [0, 4]])
    A = np.matrix(a)
    return A.I


fdd()
i = 1
x0 = np.asarray([1, 2])  # 3.00000

ans = pow(10, -6)
fd0 = fd(x0)
fdd0 = fdd()
while np.linalg.norm(fd0) > ans:
    x1 = x0 - (fd0*fdd0)
    x0 = x1
    print("次数：%s,所得的值x:%s" % (i, x1))
    i = i + 1
    fd0 = fd(x0)
    fdd0 = fdd()
else:
    print("运算结束，找到最优值！")
    print("最优值：X=%s" % x0)
