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
