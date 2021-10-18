# 第3章/1-bernoulli_pmf.py
import random
import math
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

# 第3章/2-binom_dis.py


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

# 第3章/3-poisson_pmf.py


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


# 第3章/4-uniform_dis.py


def uniform_distribution(loc=0, scale=1):
    uniform_dis = stats.uniform(loc=loc, scale=scale)
    x = np.linspace(uniform_dis.ppf(0.01),
                    uniform_dis.ppf(0.99), 100)
    fig, ax = plt.subplots(1, 1)

    # 直接传入参数
    ax.plot(x, stats.uniform.pdf(x, loc=2, scale=4), 'r-',
            lw=5, alpha=0.6, label='uniform pdf')

    # 从冻结的均匀分布取值
    ax.plot(x, uniform_dis.pdf(x), 'k-',
            lw=2, label='frozen pdf')

    # 计算ppf分别等于0.001, 0.5, 0.999时的x值
    vals = uniform_dis.ppf([0.001, 0.5, 0.999])
    print(vals)  # [ 2.004  4.     5.996]

    # 检测cdf 和 ppf的精确度
    print(np.allclose([0.001, 0.5, 0.999], uniform_dis.cdf(vals)))  # 结果为Ture

    r = uniform_dis.rvs(size=10000)
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    plt.ylabel('Probability')
    plt.title(r'PDF of Unif({}, {})'.format(loc, loc+scale))
    ax.legend(loc='best', frameon=False)
    plt.show()


uniform_distribution(loc=2, scale=4)

# 第3章/5-exponential_dis.py


def exponential_dis(loc=0, scale=1.0):
    """
    指数分布，exponential continuous random variable
    按照定义，指数分布只有一个参数lambda，这里的scale = 1/lambda
    :param loc: 定义域的左端点，相当于将整体分布沿x轴平移loc
    :param scale: lambda的倒数，loc + scale表示该分布的均值，scale^2表示该分布的方差
    :return:
    """
    exp_dis = stats.expon(loc=loc, scale=scale)
    x = np.linspace(exp_dis.ppf(0.000001),
                    exp_dis.ppf(0.999999), 100)
    fig, ax = plt.subplots(1, 1)

    # 直接传入参数
    ax.plot(x, stats.expon.pdf(x, loc=loc, scale=scale), 'r-',
            lw=5, alpha=0.6, label='uniform pdf')

    # 从冻结的均匀分布取值
    ax.plot(x, exp_dis.pdf(x), 'k-',
            lw=2, label='frozen pdf')

    # 计算ppf分别等于0.001, 0.5, 0.999时的x值
    vals = exp_dis.ppf([0.001, 0.5, 0.999])
    print(vals)  # [ 2.004  4.     5.996]

    # 检测cdf 和 ppf的精确度
    print(np.allclose([0.001, 0.5, 0.999], exp_dis.cdf(vals)))

    r = exp_dis.rvs(size=10000)
    ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
    plt.ylabel('Probability')
    plt.title(r'PDF of Exp(0.5)')
    ax.legend(loc='best', frameon=False)
    plt.show()


exponential_dis(loc=0, scale=2)

# 第3章/6-normal_dis.py
# 绘制正态分布概率密度函数

u = 0  # 均值μ
u01 = -2
sig = math.sqrt(0.2)  # 标准差δ

x = np.linspace(u - 3 * sig, u + 3 * sig, 50)
y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
print(x)
print("=" * 20)
print(y_sig)
plt.plot(x, y_sig, "r-", linewidth=2)
plt.grid(True)
plt.show()

# 第3章/7-binom_dis.py


def diff_chi2_dis():
    """
    不同参数下的卡方分布
    :return:
    """
    # chi2_dis_0_5 = stats.chi2(df=0.5)
    chi2_dis_1 = stats.chi2(df=1)
    chi2_dis_4 = stats.chi2(df=4)
    chi2_dis_10 = stats.chi2(df=10)
    chi2_dis_20 = stats.chi2(df=20)

    # x1 = np.linspace(chi2_dis_0_5.ppf(0.01), chi2_dis_0_5.ppf(0.99), 100)
    x2 = np.linspace(chi2_dis_1.ppf(0.65), chi2_dis_1.ppf(0.9999999), 100)
    x3 = np.linspace(chi2_dis_4.ppf(0.000001), chi2_dis_4.ppf(0.999999), 100)
    x4 = np.linspace(chi2_dis_10.ppf(0.000001), chi2_dis_10.ppf(0.99999), 100)
    x5 = np.linspace(chi2_dis_20.ppf(0.00000001), chi2_dis_20.ppf(0.9999), 100)
    fig, ax = plt.subplots(1, 1)
    # ax.plot(x1, chi2_dis_0_5.pdf(x1), 'b-', lw=2, label=r'df = 0.5')
    ax.plot(x2, chi2_dis_1.pdf(x2), 'g-', lw=2, label='df = 1')
    ax.plot(x3, chi2_dis_4.pdf(x3), 'r-', lw=2, label='df = 4')
    ax.plot(x4, chi2_dis_10.pdf(x4), 'b-', lw=2, label='df = 10')
    ax.plot(x5, chi2_dis_20.pdf(x5), 'y-', lw=2, label='df = 20')
    plt.ylabel('Probability')
    plt.title(r'PDF of $\chi^2$ Distribution')
    ax.legend(loc='best', frameon=False)
    plt.show()


diff_chi2_dis()

# 第3章/8-binom_dis.py


def diff_t_dis():
    """
    不同参数下的t分布
    :return:
    """
    norm_dis = stats.norm()
    t_dis_1 = stats.t(df=1)
    t_dis_4 = stats.t(df=4)
    t_dis_10 = stats.t(df=10)
    t_dis_20 = stats.t(df=20)

    x1 = np.linspace(norm_dis.ppf(0.000001), norm_dis.ppf(0.999999), 1000)
    x2 = np.linspace(t_dis_1.ppf(0.04), t_dis_1.ppf(0.96), 1000)
    x3 = np.linspace(t_dis_4.ppf(0.001), t_dis_4.ppf(0.999), 1000)
    x4 = np.linspace(t_dis_10.ppf(0.001), t_dis_10.ppf(0.999), 1000)
    x5 = np.linspace(t_dis_20.ppf(0.001), t_dis_20.ppf(0.999), 1000)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x1, norm_dis.pdf(x1), 'r-', lw=2, label=r'N(0, 1)')
    ax.plot(x2, t_dis_1.pdf(x2), 'b-', lw=2, label='t(1)')
    ax.plot(x3, t_dis_4.pdf(x3), 'g-', lw=2, label='t(4)')
    ax.plot(x4, t_dis_10.pdf(x4), 'm-', lw=2, label='t(10)')
    ax.plot(x5, t_dis_20.pdf(x5), 'y-', lw=2, label='t(20)')
    plt.ylabel('Probability')
    plt.title(r'PDF of t Distribution')
    ax.legend(loc='best', frameon=False)
    plt.show()


diff_t_dis()

# 第3章/9-binom_dis.py


def diff_f_dis():
    """
    不同参数下的F分布
    :return:
    """
#   f_dis_0_5 = stats.f(dfn=10, dfd=1)
    f_dis_1_30 = stats.f(dfn=1, dfd=30)
    f_dis_30_5 = stats.f(dfn=30, dfd=5)
    f_dis_30_30 = stats.f(dfn=30, dfd=30)
    f_dis_30_100 = stats.f(dfn=30, dfd=100)
    f_dis_100_100 = stats.f(dfn=100, dfd=100)

#   x1 = np.linspace(f_dis_0_5.ppf(0.01), f_dis_0_5.ppf(0.99), 100)
    x2 = np.linspace(f_dis_1_30.ppf(0.2), f_dis_1_30.ppf(0.99), 100)
    x3 = np.linspace(f_dis_30_5.ppf(0.00001), f_dis_30_5.ppf(0.99), 100)
    x4 = np.linspace(f_dis_30_30.ppf(0.00001), f_dis_30_30.ppf(0.999), 100)
    x6 = np.linspace(f_dis_30_100.ppf(0.0001), f_dis_30_100.ppf(0.999), 100)
    x5 = np.linspace(f_dis_100_100.ppf(0.0001), f_dis_100_100.ppf(0.9999), 100)
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
#   ax.plot(x1, f_dis_0_5.pdf(x1), 'b-', lw=2, label=r'F(0.5, 0.5)')
    ax.plot(x2, f_dis_1_30.pdf(x2), 'g-', lw=2, label='F(1, 30)')
    ax.plot(x3, f_dis_30_5.pdf(x3), 'r-', lw=2, label='F(30, 5)')
    ax.plot(x4, f_dis_30_30.pdf(x4), 'm-', lw=2, label='F(30, 30)')
    ax.plot(x6, f_dis_30_100.pdf(x6), 'c-', lw=2, label='F(30, 100)')
    ax.plot(x5, f_dis_100_100.pdf(x5), 'y-', lw=2, label='F(100, 100)')

    plt.ylabel('Probability')
    plt.title(r'PDF of f Distribution')
    ax.legend(loc='best', frameon=False)
    plt.savefig('f_diff_pdf.png', dip=500)
    plt.show()


diff_f_dis()

# 第3章/10-binom_dis.py


def flip_plot(minExp, maxExp):
    """
    Assumes minExp and maxExp positive integers; minExp < maxExp
    Plots results of 2**minExp to 2**maxExp coin flips
    """
    # 两个参数的含义，抛硬币的次数为2的minExp次方到2的maxExp次方，也就是一共做了(2**maxExp - 2**minExp)批次实验，每批次重复抛硬币2**n次

    ratios = []
    xAxis = []
    for exp in range(minExp, maxExp + 1):
        xAxis.append(2**exp)
    for numFlips in xAxis:
        numHeads = 0  # 初始化，硬币正面朝上的计数为0
        for n in range(numFlips):
            if random.random() < 0.5:  # random.random()从[0, 1)随机的取出一个数
                numHeads += 1  # 当随机取出的数小于0.5时，正面朝上的计数加1
        numTails = numFlips - numHeads  # 得到本次试验中反面朝上的次数
        ratios.append(numHeads/float(numTails))  # 正反面计数的比值
    plt.title('Heads/Tails Ratios')
    plt.xlabel('Number of Flips')
    plt.ylabel('Heads/Tails')
    plt.plot(xAxis, ratios)
    plt.hlines(1, 0, xAxis[-1], linestyles='dashed', colors='r')
    plt.show()


flip_plot(4, 16)


# 第3章/11-central_limit.py
random_data = np.random.randint(1, 7, 10000)
print random_data.mean()  # 打印平均值
print random_data.std()  # 打印标准差
sample1 = []
for i in range(0, 10):
    sample1.append(random_data[int(np.random.random() * len(random_data))])

print sample1  # 打印出来
samples = []
samples_mean = []
samples_std = []

for i in range(0, 1000):
    sample = []
    for j in range(0, 50):
        sample.append(random_data[int(np.random.random() * len(random_data))])
    sample_np = np.array(sample)
    samples_mean.append(sample_np.mean())
    samples_std.append(sample_np.std())
    samples.append(sample_np)

samples_mean_np = np.array(samples_mean)
samples_std_np = np.array(samples_std)

print samples_mean_np
