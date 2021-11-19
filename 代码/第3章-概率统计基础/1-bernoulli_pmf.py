# 第3章/bernoulli_pmf.py


# NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，
# 支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。

# NumPy 的前身 Numeric 最早是由 Jim Hugunin 与其它协作者共同开发，
# 2005 年，Travis Oliphant 在 Numeric 中结合了另一个同性质的程序库 Numarray 的特色，
# 并加入了其它扩展而开发了 NumPy。NumPy 为开放源代码并且由许多协作者共同维护开发。

# NumPy 是一个运行速度非常快的数学库，主要用于数组计算，包含：
# 一个强大的N维数组对象 ndarray
# 广播功能函数
# 整合 C/C++/Fortran 代码的工具
# 线性代数、傅里叶变换、随机数生成等功能


# SciPy 是一个开源的 Python 算法库和数学工具包。
# Scipy 是基于 Numpy 的科学计算库，
# 用于数学、科学、工程学等领域，很多有一些高阶抽象和物理模型需要使用 Scipy。
# SciPy 包含的模块有最优化、线性代数、积分、插值、
# 特殊函数、快速傅里叶变换、信号处理和图像处理、
# 常微分方程求解和其他科学与工程中常用的计算。


# Matplotlib 是 Python 的绘图库。
# 它可与 NumPy 一起使用，提供了一种有效的 MatLab 开源替代方案。
# 它也可以和图形工具包一起使用，如 PyQt 和 wxPython。

# 在概率论中，
# 概率质量函数（probability mass function，简写作pmf）是离散随机变量在各特定取值上的概率。
# 概率质量函数和概率密度函数不同之处在于：
# 概率质量函数是对离散随机变量定义的，本身代表该值的概率；
# 概率密度函数本身不是概率，只有对连续随机变量的概率密度函数在某区间内进行积分后才是概率。

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def bernoulli_pmf(p=0.0):
    ber_dist = stats.bernoulli(p)
    x = [0, 1]
    x_name = ['0', '1']
    pmf = [ber_dist.pmf(x[0]), ber_dist.pmf(x[1])]
    plt.bar(x, pmf, width=0.15)
    plt.xticks(x, x_name)
    plt.ylabel('Probability')
    plt.title('PMF of bernoulli distribution')
    plt.show()


bernoulli_pmf(p=0.3)
