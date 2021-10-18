
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
