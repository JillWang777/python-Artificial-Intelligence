import random
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
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
