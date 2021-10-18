#9.1 曲线拟合实验
#导入相应的Python包和模块
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import pylab

#定义draw_sin_line() 函数，该函数用来绘制标准的sin曲线
def draw_sin_line():

#绘制标准的sin曲线
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
weights = tf.get_variable("weight", [1, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
biases = tf.get_variable("bias", [1, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
hidden1 = tf.sigmoid(tf.multiply(input_data, weights) + biases)
with tf.variable_scope('hidden2'):
# 第2个隐藏层，采用16个隐藏节点
weights = tf.get_variaible("weight", [16, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
biases = tf.get_variable("bias", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
mul = tf.matmul(hidden1, weights)
hidden2 = tf.sigmoid(mul + biases)
with tf.variable_scope('hidden3'):
# 第3个隐藏层，采用16个隐藏节点
weights = tf.get_variaaible("weight", [16, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
biases = tf.get_variaiable("bias", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
mul = tf.matmul(hidden2, weights)
hidden3 = tf.sigmoid(mul + biases)
with tf.variable_scope('output_layer'):
# 输出层
weights = tf.get_variaible("weight", [16, 1], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
biases = tf.get_variable("bias", [1], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
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
pylab.plot(test_x_ndarray, test_y_ndarray, '--', label = str(times) + ' times')
pylab.legend(loc='upper right')
pylab.show()
print("=== DONE ===")

-----------------------------------------------------------------------------------------------------
#9.2 泰坦尼克号乘客死亡概率预测
#!/usr/bin/env python 
# -*- coding:utf-8 -*-


import numpy as np
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

# Preprocessing function
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # Converting 'sex' field to float (id is 1 after removing labels column)
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)


# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)


# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# 将对应人员数据存入对应的list内
dicaprio = [3,"Mr. Bernt",'male',0,0,0,65306,8.1125]
winslet = [1,"Allen, Miss. Elisabeth Walton",'female',29,0,0,24160,211.3375]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)


# 预测对应人员生还率
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])


-----------------------------------------------------------------------------------------------------
#9.3 股票预测
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv(‘data_stocks.csv’)
data.describe()
data.info()
data.head()
print(time.strftime('%Y-%m-%d', time.localtime(data['DATE'].max())),
      time.strftime('%Y-%m-%d', time.localtime(data['DATE'].min())))
plt.plot(data['SP500'])
data.drop('DATE', axis=1, inplace=True)
data_train = data.iloc[:int(data.shape[0] * 0.8), :]
data_test = data.iloc[int(data.shape[0] * 0.8):, :]
print(data_train.shape, data_test.shape)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

input_dim = X_train.shape[1]
hidden_1 = 1024
hidden_2 = 512
hidden_3 = 256
hidden_4 = 128
output_dim = 1
batch_size = 256
epochs = 10

tf.reset_default_graph()

X = tf.placeholder(shape=[None, input_dim], dtype=tf.float32)
Y = tf.placeholder(shape=[None], dtype=tf.float32)

W1 = tf.get_variable('W1', [input_dim, hidden_1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable('b1', [hidden_1], initializer=tf.zeros_initializer())
W2 = tf.get_variable('W2', [hidden_1, hidden_2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b2 = tf.get_variable('b2', [hidden_2], initializer=tf.zeros_initializer())
W3 = tf.get_variable('W3', [hidden_2, hidden_3], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b3 = tf.get_variable('b3', [hidden_3], initializer=tf.zeros_initializer())
W4 = tf.get_variable('W4', [hidden_3, hidden_4], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b4 = tf.get_variable('b4', [hidden_4], initializer=tf.zeros_initializer())
W5 = tf.get_variable('W5', [hidden_4, output_dim], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b5 = tf.get_variable('b5', [output_dim], initializer=tf.zeros_initializer())

h1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2))
h3 = tf.nn.relu(tf.add(tf.matmul(h2, W3), b3))
h4 = tf.nn.relu(tf.add(tf.matmul(h3, W4), b4))
out = tf.transpose(tf.add(tf.matmul(h4, W5), b5))

cost = tf.reduce_mean(tf.squared_difference(out, Y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        shuffle_indices = np.random.permutation(np.arange(y_train.shape[0]))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        for i in range(y_train.shape[0] // batch_size):
            start = i * batch_size
            batch_x = X_train[start : start + batch_size]
            batch_y = y_train[start : start + batch_size]
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            if i % 50 == 0:
                print('MSE Train:', sess.run(cost, feed_dict={X: X_train, Y: y_train}))
                print('MSE Test:', sess.run(cost, feed_dict={X: X_test, Y: y_test}))
                y_pred = sess.run(out, feed_dict={X: X_test})
                y_pred = np.squeeze(y_pred)
                plt.plot(y_test, label='test')
                plt.plot(y_pred, label='pred')
                plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
                plt.legend()
                plt.show()
from keras.layers import Input, Dense
from keras.models import Model

X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

input_dim = X_train.shape[1]
hidden_1 = 1024
hidden_2 = 512
hidden_3 = 256
hidden_4 = 128
output_dim = 1
batch_size = 256
epochs = 10

X = Input(shape=[input_dim,])
h = Dense(hidden_1, activation='relu')(X)
h = Dense(hidden_2, activation='relu')(h)
h = Dense(hidden_3, activation='relu')(h)
h = Dense(hidden_4, activation='relu')(h)
Y = Dense(output_dim, activation='sigmoid')(h)

model = Model(X, Y)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
y_pred = model.predict(X_test)
print('MSE Train:', model.evaluate(X_train, y_train, batch_size=batch_size))
print('MSE Test:', model.evaluate(X_test, y_test, batch_size=batch_size))
plt.plot(y_test, label='test')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()
from keras.layers import Input, Dense, LSTM
from keras.models import Model

output_dim = 1
batch_size = 256
epochs = 10
seq_len = 5
hidden_size = 128

X_train = np.array([data_train[i : i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])[:, :, np.newaxis]
y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])
X_test = np.array([data_test[i : i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])[:, :, np.newaxis]
y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X = Input(shape=[X_train.shape[1], X_train.shape[2],])
h = LSTM(hidden_size, activation='relu')(X)
Y = Dense(output_dim, activation='sigmoid')(h)

model = Model(X, Y)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
y_pred = model.predict(X_test)
print('MSE Train:', model.evaluate(X_train, y_train, batch_size=batch_size))
print('MSE Test:', model.evaluate(X_test, y_test, batch_size=batch_size))
plt.plot(y_test, label='test')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()

-----------------------------------------------------------------------------------------------------
#9.4 车牌识别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'是为了解决CPU使用tensorflow不能兼容AVX2指令的问题
#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#本实验使用了tkinter库，tkinter是Python自带的可用于GUI编程的库
#利用tkinter可以建立一个选择车牌图片的对话框
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

#生成对话框，选择车牌图片，将路径和文件名以字符串类型返回给filename，并打印出来
filename = filedialog.askopenfilename()
print(filename)

#使用hyperlpr库识别刚才选择的车牌图片
from hyperlpr import pipline as  pp
import cv2
image = cv2.imread(filename)
image,res  = pp.SimpleRecognizePlateByE2E(image)

#将识别的结果输出
print(res)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'是为了解决CPU使用tensorflow不能兼容AVX2指令的问题
#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import cv2
from hyperlpr import pipline as pp
import click
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

#生成对话框，选择车牌图片，将路径和文件名以字符串类型返回给filename，并打印出来
filename = filedialog.askopenfilename()
print(filename)

@click.command()
@click.option('--video', help = 'input video file')
def main(video):
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()
    stream = cv2.VideoCapture(filename)
    time.sleep(2.0)

    while True:
        # grab the frame from the threaded video stream
        grabbed, frame = stream.read()
        if not grabbed:
            print('No data, break.')
            break

        _, res = pp.SimpleRecognizePlate(frame)

        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rgb = imutils.resize(frame, width = 750)
        # r = frame.shape[1] / float(rgb.shape[1])

        cv2.putText(frame, str(res), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    stream.release()


if __name__ == '__main__':
    main()

-----------------------------------------------------------------------------------------------------
#9.5 口罩识别
# -*- coding:utf-8 -*-
import cv2
import time
import argparse

import numpy as np
from PIL import Image
from keras.models import model_from_json
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference


#导入口罩智能识别模型
sess, graph = load_tf_model('models\face_mask_detection.pb')
# anchor configuration锚点设置
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors生成锚点
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
def inference(image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160), draw_result=True, show_result=True):
    '''  检测推理的主要功能
   # ：param image：3D numpy图片数组
    #  ：param conf_thresh：分类概率的最小阈值。
   #  ：param iou_thresh：网管的IOU门限
   #  ：param target_shape：模型输入大小。
   #  ：param draw_result：是否将边框拖入图像。
   #  ：param show_result：是否显示图像。
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # 为了加快速度，请执行单类NMS，而不是多类NMS。
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx是nms之后的活动边界框。
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh,iou_thresh=iou_thresh)
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # 裁剪坐标，避免该值超出图像边界。
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info
def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(260, 260),
                      draw_result=True,
                      show_result=False)
            cv2.imshow('image', img_raw[:, :, ::-1])
            cv2.waitKey(1)
            inference_stamp = time.time()
            # writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp, inference_stamp - read_frame_stamp, write_frame_stamp - inference_stamp))
#1.检测图片中是否佩戴口罩
#检测图片放在img目录中，里面存放有测试图片进行测试
img = cv2.imread("img/test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#执行推断程序，并将结果输出
inference(img, show_result=True, target_shape=(260, 260))
#2.检测监控视频哪些人群佩戴口罩，哪些人群没有配到口罩
#参数mp4/test.mp4表示
run_on_video("mp4/test.mp4", '', conf_thresh=0.5)

#github网址
github.com/zlanngao/deeplearning

