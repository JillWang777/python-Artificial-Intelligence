# 4-交叉熵损失函数的Python代码实现


def binary_crossentropy(t, o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))
