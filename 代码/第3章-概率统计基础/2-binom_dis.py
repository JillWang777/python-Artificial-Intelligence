# 第3章/2-binom_dis.py
import random
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def binom_dis(n=1, p=0.1):
    binom_dis = stats.binom(n, p)
    x = np.arange(binom_dis.ppf(0.0001), binom_dis.ppf(0.9999))
    print(x)  # [ 0.  1.  2.  3.  4.]
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, binom_dis.pmf(x), 'bo', label='binom pmf')
    ax.vlines(x, 0, binom_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('PMF of binomial distribution(n={}, p={})'.format(n, p))
    plt.show()


binom_dis(n=20, p=0.6)
