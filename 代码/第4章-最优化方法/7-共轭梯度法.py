
# 7、共轭梯度法Python代码实现


def fd(x):
    t = np.asarray([1, 2])
    #y = np.dot(x.T,t)
    y = x.T * t
    return y


Q = np.asarray([[1, 0], [0, 2]])
x0 = np.asarray([2, 1])
fd0 = fd(x0)
d0 = -fd(x0)
a0 = -np.dot(fd0.T, d0)/np.dot(np.dot(d0.T, Q), d0)
x1 = x0+np.dot(a0, d0)
ans = pow(10, -6)
fd1 = fd(x1)
while np.linalg.norm(fd1) > ans:
    b0 = pow(np.linalg.norm(fd1), 2)/pow(np.linalg.norm(fd0), 2)
    d1 = -fd1+np.dot(b0, d0)
    a1 = -np.dot(fd1.T, d1)/np.dot(np.dot(d1.T, Q), d1)
    x2 = x1+np.dot(a1, d1)
    x0 = x1
    x1 = x2
    fd1 = fd(x1)
    fd0 = fd(x0)
    d0 = -fd(x0)
print("最优值：", x1)
