# 5、梯度下降法的Python实现


def gradient_decent(fn, partial_derivatives, n_variables, lr=0.1,
                    max_iter=10000, tolerance=1e-5):
    theta = [random() for _ in range(n_variables)]
    y_cur = fn(*theta)
    for i in range(max_iter):
        # Calculate gradient of current theta.
        gradient = [f(*theta) for f in partial_derivatives]
        # Update the theta by the gradient.
        for j in range(n_variables):
            theta[j] -= gradient[j] * lr
        # Check if converged or not.
        y_cur, y_pre = fn(*theta), y_cur
        if abs(y_pre - y_cur) < tolerance:
            break
    return theta, y_cur


def f(x, y):
    return (x + y - 3) ** 2 + (x + 2 * y - 5) ** 2 + 2


def df_dx(x, y):
    return 2 * (x + y - 3) + 2 * (x + 2 * y - 5)


def df_dy(x, y):
    return 2 * (x + y - 3) + 4 * (x + 2 * y - 5)


def main():
    print("Solve the minimum value of quadratic function:")
    n_variables = 2
    theta, f_theta = gradient_decent(f, [df_dx, df_dy], n_variables)
    theta = [round(x, 3) for x in theta]
    print("The solution is: theta %s, f(theta) %.2f.\n" % (theta, f_theta))
