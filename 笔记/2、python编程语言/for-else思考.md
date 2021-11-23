> 老婆给当程序员的老公打电话：“下班顺路买一斤包子带回来，如果看到卖西瓜的，买一个。”当晚，程序员老公手捧一个包子进了家门……老婆怒道：“你怎么就买了一个包子？”老公答曰：“因为看到了卖西瓜的。”

程序员买西瓜的笑话可能大部分读者都知道，今天写的这篇文章和这个笑话有一定的关系。

任何编程语言都提供了 if...else... 语句，表示如果（if）满足条件就做某件事，否则（else）就做另外一件事：

```
if a==b:
    print("true")

else:
    print("false")
```

然而，在 Python 中 else 不仅可以和 if 搭配使用，还有另一种特有的句法是 for…else …，除此之外，它还可以和 while、try…except 组合使用，例如：

```
for i in range(3):
    print(i)
else:
    print("end")

>>>
0
1
2
end
```

但是，你会发现 for…else… 与 if…else… 表现得不一样，按照以往经验来说，执行了 for 语句块中的代码就不执行 else 里面的，反之亦然。

然而，我们看到的却恰恰相反，for 循环结束之后接着又执行了 else 语句块，这就有点意思了，if … else … 翻译成白话就是 **如果…否则…**，而 for…else… 翻译成白话成了 **直到… 然后 …**，为什么不把它写成 for…then… 的句式呢？这不更好理解吗？

另外，即使 for 循环遍历的是一个空列表也会执行 else 语句块。

```
for i in []:
    print(i)
else:
    print("end")

>>>
end
```

继续探索，如果我们用 `break` 提前终止 for 循环，会发生什么？

```
for i in range(3):
    print(i)
    if i % 2 == 0:
        break
else:
    print("end")

>>>
0
```

循环遇到 break 退出后，整个语句就结束，else 语句块也不执行了。

综上，我们可以得出这样一个结论，只有当循环里没有遇到 break 时，else 块才会执行。此刻，你应该明白了，真正和 else 搭配使用的是 for 循环中的 break，break ... else ... 才是两个互斥的条件

Python 之父为什么要搞出这样的一种语法糖出来呢？这是我们常人没法理解的。不过「python之禅」告诉了我们答案： *"Although that way may not be obvious at first unless you're Dutch."*。

在平时的开发中真的很少有 for...else... 的应用场景，不过，像下面这种场景用 for else 还真是一种 pythonic 的用法。

当你用 for 循环迭代查找列表的中的某个元素时，如果找到了就立刻退出循环，如果迭代完了列表还没找到需要以另外一种形式（比如异常）的方式通知调用者时，用 for...else... 无疑是最好的选择。

```
# https://stackoverflow.com/a/9980752/1392860
for i in mylist:
    if i == target:
        break
    process(i)
else:
    raise ValueError("List argument missing terminal flag.")
```

如果不用 for...else... ， 那么还需要专门建立一个临时标记变量来标记是否已经找到了

```
found = False
for i in mylist:
    if i == target:
        found = True
        break
    process(i)

if not found:
    raise ValueError("List argument missing terminal flag.")
```

当你想在房间里找某样东西时，只要在任意位置找到了，就停止继续搜查工作。但如果把整个房间都翻遍了，还没找到我们想要的东西时，需要告诉人家，很抱歉，这儿没有你要找的东西。遇到这样的情况时就用 for ... else ，除此之外，恐怕只会引起误操作。

参考链接：https://stackoverflow.com/questions/9979970/why-does-python-use-else-after-for-and-while-loops/9980752#9980752