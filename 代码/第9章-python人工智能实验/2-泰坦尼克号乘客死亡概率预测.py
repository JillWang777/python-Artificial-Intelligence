# 9.2 泰坦尼克号乘客死亡概率预测
#!/usr/bin/env python
# -*- coding:utf-8 -*-


from tflearn.data_utils import load_csv
import numpy as np
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
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
to_ignore = [1, 6]

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
dicaprio = [3, "Mr. Bernt", 'male', 0, 0, 0, 65306, 8.1125]
winslet = [1, "Allen, Miss. Elisabeth Walton",
           'female', 29, 0, 0, 24160, 211.3375]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)


# 预测对应人员生还率
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])
