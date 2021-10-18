# 2、全局最优化


def fo(*args):
    x = args[0][0]
    y = args[0][1]
    z = np.sin(x) + 0.05*x**2 + np.sin(y) + 0.05*y**2
    # print(x,y,z)
    return z


opt = spo.brute(fo, ((-10, 10, 0.1), (-10, 10, 0.1)), finish=None)
print(opt)
print(fm(opt[0], opt[1]))
