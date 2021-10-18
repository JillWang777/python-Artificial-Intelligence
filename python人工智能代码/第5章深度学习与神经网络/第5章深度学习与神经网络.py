# 1-正向传播函数
import numpy as np


def _feedForward(self, keep_prob):
    """ Forward pass """
    z = []
    a = []
    z.append(np.dot(self.W[0], self.X) + self.B[0])
    a.append(PHI[self.ac_funcs[0]](z[-1]))
    for l in range(1, len(self.layers)):
    z.append(np.dot(self.W[l], a[-1]) + self.B[l])
    # a.append(PHI[self.ac_funcs[l]](z[l]))
    _a = PHI[self.ac_funcs[l]](z[l])
    a.append(((np.random.rand(_a.shape[0], 1) < keep_prob)*_a)/keep_prob)
    return z, a

# 2-反向传播函数


def startTraining(self, epochs, alpha, _lambda, keep_prob=0.5, interval=100):
    """
    Start training the neural network. It takes the followng parameters :  
    1. epochs : 训练网络的迭代次数 
    2. alpha : 学习率 
    3. lambda : L2正则化参数或惩罚参数 
    4. keep_prob : 丢弃正则化参数，意味部分神经元失活 
    5. interval :误差和精度更新的间隔
    """
    start = time.time()
    for i in range(epochs+1):
    z, a = self._feedForward(keep_prob)
    delta = self._cost_derivative(a[-1])
    for l in range(1, len(z)):
    delta_w = np.dot(delta, a[-l-1].T) + (_lambda)*self.W[-l]
    delta_b = np.sum(delta, axis=1, keepdims=True)
    delta = np.dot(self.W[-l].T, delta)*PHI_PRIME[self.ac_funcs[-l-1]](z[-l-1])
    self.W[-l] = self.W[-l] - (alpha/self.m)*delta_w
    self.B[-l] = self.B[-l] - (alpha/self.m)*delta_b
    delta_w = np.dot(delta, self.X.T) + (_lambda)*self.W[0]
    delta_b = np.sum(delta, axis=1, keepdims=True)
    self.W[0] = self.W[0] - (alpha/self.m)*delta_w
    self.B[0] = self.B[0] - (alpha/self.m)*delta_b


if not i % interval:
    aa = self.predict(self.X)
    if self.loss == 'b_ce':
    aa = aa > 0.5
    self.acc = sum(sum(aa == self.y)) / self.m
    cost_val = self._cost_func(a[-1], _lambda)
    self.cost.append(cost_val)
    elif self.loss == 'c_ce':
    aa = np.argmax(aa, axis=0)
    yy = np.argmax(self.y, axis=0)
    self.acc = np.sum(aa == yy)/(self.m)
    cost_val = self._cost_func(a[-1], _lambda)
    self.cost.append(cost_val)
    sys.stdout.write(
        f'\rEpoch[{i}] : Cost = {cost_val:.2f} ; Acc = {(self.acc*100):.2f}% ; Time Taken = {(time.time()-start):.2f}s')
    print('\n')
    return None

# 3-激活函数python实现


def sigmoid(z):
    """ Reutrns the element wise sigmoid function. """
    return 1./(1 + np.exp(-z))


def sigmoid_prime(z):
    """ Returns the derivative of the sigmoid function. """
    return sigmoid(z)*(1-sigmoid(z))


def ReLU(z):
    """ Reutrns the element wise ReLU function. """
    return (z*(z > 0))


def ReLU_prime(z):
    """ Returns the derivative of the ReLU function. """
    return 1*(z >= 0)


def lReLU(z):
    """ Reutrns the element wise leaky ReLU function. """
    return np.maximum(z/100, z)


def lReLU_prime(z):
    """ Returns the derivative of the leaky ReLU function. """
    z = 1*(z >= 0)
    z[z == 0] = 1/100
    return z


def tanh(z):
    """ Reutrns the element wise hyperbolic tangent function. """
    return np.tanh(z)


def tanh_prime(z):
    """ Returns the derivative of the tanh function. """
    return (1-tanh(z)**2)

# 4-交叉熵损失函数的Python代码实现


def binary_crossentropy(t, o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

# 5-Softmax函数的Python实现代码


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# 6-简单循环神经网络前向传播

X = [1, 2]
state = [0.0, 0.0]
# 分开定义不同输入部分的权重以方便操作
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# 定义用于输出的全连接层参数
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 执行前向传播过程
for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + \
        X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)
    final_output = np.dot(state, w_output) + b_output
    print("before activation: ", before_activation)
    print("state: ", state)
    print("output: ", final_output)

# 7-LSTM前向推导过程的Python实现代码


def bottom_data_is(self, x, s_prev=None, h_prev=None):
    # if this is the first lstm node in the network
    if s_prev == None:
        s_prev = np.zeros_like(self.state.s)
    if h_prev == None:
        h_prev = np.zeros_like(self.state.h)
    # save data for use in backprop
    self.s_prev = s_prev
    self.h_prev = h_prev

    # concatenate x(t) and h(t-1)
    xc = np.hstack((x,  h_prev))
    self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
    self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
    self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
    self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
    self.state.s = self.state.g * self.state.i + s_prev * self.state.f
    self.state.h = self.state.s * self.state.o
    self.x = x
    self.xc = xc


# 8-LSTM长短时记忆网络的Python实现代码
def y_list_is(self, y_list, loss_layer):
    """
    Updates diffs by setting target sequence
    with corresponding loss layer.
    Will *NOT* update parameters.  To update parameters,
    call self.lstm_param.apply_diff()
    """
    assert len(y_list) == len(self.x_list)
    idx = len(self.x_list) - 1
    # first node only gets diffs from label ...
    loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
    diff_h = loss_layer.bottom_diff(
        self.lstm_node_list[idx].state.h, y_list[idx])
    # here s is not affecting loss due to h(t+1), hence we set equal to zero
    diff_s = np.zeros(self.lstm_param.mem_cell_ct)
    self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
    idx -= 1

    # ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
    # we also propagate error along constant error carousel using diff_s
    while idx >= 0:
        loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(
            self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
        diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1

    return loss


def bottom_diff(self, pred, label):
    diff = np.zeros_like(pred)
    diff[0] = 2 * (pred[0] - label)
    return diff


def top_diff_is(self, top_diff_h, top_diff_s):
    # notice that top_diff_s is carried along the constant error carousel
    ds = self.state.o * top_diff_h + top_diff_s
    do = self.state.s * top_diff_h
    di = self.state.g * ds
    dg = self.state.i * ds
    df = self.s_prev * ds

    # diffs w.r.t. vector inside sigma / tanh function
    di_input = (1. - self.state.i) * self.state.i * di  # sigmoid diff
    df_input = (1. - self.state.f) * self.state.f * df
    do_input = (1. - self.state.o) * self.state.o * do
    dg_input = (1. - self.state.g ** 2) * dg  # tanh diff

    # diffs w.r.t. inputs
    self.param.wi_diff += np.outer(di_input, self.xc)
    self.param.wf_diff += np.outer(df_input, self.xc)
    self.param.wo_diff += np.outer(do_input, self.xc)
    self.param.wg_diff += np.outer(dg_input, self.xc)
    self.param.bi_diff += di_input
    self.param.bf_diff += df_input
    self.param.bo_diff += do_input
    self.param.bg_diff += dg_input

    # compute bottom diff
    dxc = np.zeros_like(self.xc)
    dxc += np.dot(self.param.wi.T, di_input)
    dxc += np.dot(self.param.wf.T, df_input)
    dxc += np.dot(self.param.wo.T, do_input)
    dxc += np.dot(self.param.wg.T, dg_input)

    # save bottom diffs
    self.state.bottom_diff_s = ds * self.state.f
    self.state.bottom_diff_x = dxc[:self.param.x_dim]
    self.state.bottom_diff_h = dxc[self.param.x_dim:]


wi_diff += np.outer((1.-i)*i*di, xc)
wf_diff += np.outer((1.-i)*i*df, xc)
wo_diff += np.outer((1.-i)*i*do, xc)
wg_diff += np.outer((1.-i)*i*dg, xc)
