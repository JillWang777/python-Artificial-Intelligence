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
