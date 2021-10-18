# 第3章/3-poisson_pmf.py

import random
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def poisson_pmf(mu=3):
    poisson_dis = stats.poisson(mu)
    x = np.arange(poisson_dis.ppf(0.001), poisson_dis.ppf(0.999))
    print(x)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, poisson_dis.pmf(x), 'bo', ms=8, label='poisson pmf')
    ax.vlines(x, 0, poisson_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('PMF of poisson distribution(mu={})'.format(mu))
    plt.show()


poisson_pmf(mu=8)
