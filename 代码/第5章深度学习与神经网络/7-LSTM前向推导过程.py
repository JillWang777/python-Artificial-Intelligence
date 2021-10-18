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
