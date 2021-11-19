class Fridge():
    No = 0
    Num = 0

    def __init__(self):
        Fridge.No += 1

    def open(self):
        print('打开%d号冰箱门' % self.No)

    def pack(self, goods):
        self.Num += 1
        self.goods = goods
        print('在%d号冰箱门装入%d个物品%s' % (self.No, self.Num, goods))

    def close(self):
        print('关上%d号冰箱门' % self.No)


class Double_Door(Fridge):
    def __init__(self):
        super().__init__()
        print('这是一个双开门冰箱')


# fridge1 = Fridge()
# fridge1.open()
# fridge1.pack('大象')
# fridge1.pack('小象')
# fridge1.pack('🐘')
# fridge1.close()

fridge_double_door = Double_Door()
fridge_double_door.pack('大象🐘')
