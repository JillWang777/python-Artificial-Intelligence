# 4、最小二乘法的Python实现
# 拟合函数


def func(a, x):
    k, b = a
    return k * x + b
# 残差


def dist(a, x, y):
    return func(a, x) - y


font = FontProperties()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']  # 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.figure()
plt.title(u' 女生的身高体重数据 ')
plt.xlabel(u'x 体重')
plt.ylabel(u'y 身高')
plt.axis([40, 80, 140, 200])
plt.grid(True)
x = np.array([48.0, 57.0, 50.0, 54.0, 64.0, 61.0, 43.0, 59.0])
y = np.array([165.0, 165.0, 157.0, 170.0, 175.0, 165.0, 155.0, 170.0])
plt.plot(x, y, 'k.')
param = [0, 0]
var = leastsq(dist, param, args=(x, y))
k, b = var[0]
print(k, b)
plt.plot(x, k*x+b, 'o-')
plt.show()
