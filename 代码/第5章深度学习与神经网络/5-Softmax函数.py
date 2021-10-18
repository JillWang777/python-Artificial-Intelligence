# 5-Softmax函数的Python实现代码


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)
