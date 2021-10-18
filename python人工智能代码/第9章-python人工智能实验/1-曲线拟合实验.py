# 9.1 曲线拟合实验
# 导入相应的Python包和模块
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import pylab

# 定义draw_sin_line() 函数，该函数用来绘制标准的sin曲线


def draw_sin_line():

    # 绘制标准的sin曲线
x = np.arange(0, 2*np.pi, 0.01)
x = x.reshape((len(x), 1))
y = np.sin(x)
pylab.plot(x, y, label='标准的sin曲线')
plt.axhline(linewidth=1, color='r')
plt.axvline(x=np.pi, linestyle='--', linewidth=1, color='g')
def get_train_data():


'''返回一个训练样本(train_x, train_y)其中 train_x 是随机的自变量，train_y 是train_x 的sin函数值''
train_x = np.random.uniform(0.0, 2*np.pi, (1))
train_y = np.sin(train_x)
return train_x, train_y
def inference(input_data):
'''定义前向计算的网络结构，args: 输入x的值，单个值'
with tf.variable_scope('hidden1'):
    # 第1个隐藏层，采用16个隐藏节点
weights = tf.get_variable(
    "weight", [1, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
biases = tf.get_variable(
    "bias", [1, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
hidden1 = tf.sigmoid(tf.multiply(input_data, weights) + biases)
with tf.variable_scope('hidden2'):
    # 第2个隐藏层，采用16个隐藏节点
weights = tf.get_variaible(
    "weight", [16, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
biases = tf.get_variable(
    "bias", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
mul = tf.matmul(hidden1, weights)
hidden2 = tf.sigmoid(mul + biases)
with tf.variable_scope('hidden3'):
    # 第3个隐藏层，采用16个隐藏节点
weights = tf.get_variaaible(
    "weight", [16, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
biases = tf.get_variaiable(
    "bias", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
mul = tf.matmul(hidden2, weights)
hidden3 = tf.sigmoid(mul + biases)
with tf.variable_scope('output_layer'):
    # 输出层
weights = tf.get_variaible(
    "weight", [16, 1], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
biases = tf.get_variable(
    "bias", [1], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
output = tf.matmul(hidden3, weights) + biases
return output


def train():


    # 学习率
learning_rate = 0.01
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# 基于训练好的模型推理，获取推理结果
net_out = inference(x)
# 定义损失函数的op
loss_op = tf.square(net_out - y)
# 采用随机梯度下降的优化函数
opt = tf.train.GradientDescentOptimizer(learning_rate)
# 定义训练操作
train_op = opt.minimize(loss_op)
# 变量初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 执行变量的初始化操作
sess.run(init)
print("开始训练 ...")
for i in range(100001):
    # 获取训练数据
train_x, train_y = get_train_data()
sess.run(train_op, feed_dict={x: train_x, y: train_y})
# 定时输出当前的状态
if i % 10000 == 0:
times = int(i/10000)
# 每执行10000次训练后，测试一下结果，测试结果用 pylab.plot()函数在界面上绘制出来
test_x_ndarray = np.arange(0, 2 * np.pi, 0.01)
test_y_ndarray = np.zeros([len(test_x_ndarray)])
ind = 0
for test_x in test_x_ndarray:
test_y = sess.run(net_out, feed_dict={x: test_x, y: 1})
# 对数组中指定的索引值指向的元素替换成指定的值
np.put(test_y_ndarray, ind, test_y)
ind += 1
# 先绘制标准的正弦函数的曲线，再用虚线绘制出模拟正弦函数的曲线
draw_sin_line()
pylab.plot(test_x_ndarray, test_y_ndarray, '--', label=str(times) + ' times')
pylab.legend(loc='upper right')
pylab.show()
print("=== DONE ===")
