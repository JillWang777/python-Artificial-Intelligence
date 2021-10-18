# 第3章/bernoulli_pmf.py
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
