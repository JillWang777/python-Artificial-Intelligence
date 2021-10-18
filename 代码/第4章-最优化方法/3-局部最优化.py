
# 3、局部最优化
opt2 = spo.fmin(fo, (2.0, 2.0), maxiter=250)
print(opt2)
print(fm(opt2[0], opt2[1]))
