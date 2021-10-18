import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
# 先建立训练集，产生100组满足y=0.1x+0.3的[x,y]集合
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

#----------------------------------------------------------
#以下程序就是利用这100个数据集[x,y]，来推算k和b值，即下面的W和b值
# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
# 初始化值：W为【-0.1,0.1]，b为全0
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
# 最小二乘法和梯度下降法
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
#以上定义好数据和解决问题的方法以后，下面就是委托tensorflow执行
#init 让tf进行初始化
init = tf.global_variables_initializer()

# Launch the graph.
# 跟tf建立对话，并委托tf执行计算
sess = tf.Session()
sess.run(init)

# Fit the line.
# 让tf将算法执行200次，每20次打印计算结果。
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
