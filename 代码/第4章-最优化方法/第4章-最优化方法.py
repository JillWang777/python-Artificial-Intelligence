# 1、效用函数优化的Python实现
from random import random
from scipy import sparse
from sklearn.linear_model import LinearRegression
from scipy.optimize import leastsq
from matplotlib.font_manager import FontProperties
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as spo


def fm(*args):
    return (np.sin(args[0])+0.05*args[0]**2+np.sin(args[1])+0.05*args[1]**2)


x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
x, y = np.meshgrid(x, y)
z = fm(x, y)

fig = plt.figure(figsize=(9, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, z, rstride=2, cstride=2,
                       cmap=mpl.cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

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

# 3、局部最优化
opt2 = spo.fmin(fo, (2.0, 2.0), maxiter=250)
print(opt2)
print(fm(opt2[0], opt2[1]))

# 4、最小二乘法的Python实现
# 拟合函数


def func(a, x):
    k, b = a
    return k * x + b
# 残差


def dist(a, x, y):
    return func(a, x) - y


font = FontProperties()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']  # 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.figure()
plt.title(u' 女生的身高体重数据 ')
plt.xlabel(u'x 体重')
plt.ylabel(u'y 身高')
plt.axis([40, 80, 140, 200])
plt.grid(True)
x = np.array([48.0, 57.0, 50.0, 54.0, 64.0, 61.0, 43.0, 59.0])
y = np.array([165.0, 165.0, 157.0, 170.0, 175.0, 165.0, 155.0, 170.0])
plt.plot(x, y, 'k.')
param = [0, 0]
var = leastsq(dist, param, args=(x, y))
k, b = var[0]
print(k, b)
plt.plot(x, k*x+b, 'o-')
plt.show()

# 5、梯度下降法的Python实现


def gradient_decent(fn, partial_derivatives, n_variables, lr=0.1,
                    max_iter=10000, tolerance=1e-5):
    theta = [random() for _ in range(n_variables)]
    y_cur = fn(*theta)
    for i in range(max_iter):
        # Calculate gradient of current theta.
        gradient = [f(*theta) for f in partial_derivatives]
        # Update the theta by the gradient.
        for j in range(n_variables):
            theta[j] -= gradient[j] * lr
        # Check if converged or not.
        y_cur, y_pre = fn(*theta), y_cur
        if abs(y_pre - y_cur) < tolerance:
            break
    return theta, y_cur


def f(x, y):
    return (x + y - 3) ** 2 + (x + 2 * y - 5) ** 2 + 2


def df_dx(x, y):
    return 2 * (x + y - 3) + 2 * (x + 2 * y - 5)


def df_dy(x, y):
    return 2 * (x + y - 3) + 4 * (x + 2 * y - 5)


def main():
    print("Solve the minimum value of quadratic function:")
    n_variables = 2
    theta, f_theta = gradient_decent(f, [df_dx, df_dy], n_variables)
    theta = [round(x, 3) for x in theta]
    print("The solution is: theta %s, f(theta) %.2f.\n" % (theta, f_theta))


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
