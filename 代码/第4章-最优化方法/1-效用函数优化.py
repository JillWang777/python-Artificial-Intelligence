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
