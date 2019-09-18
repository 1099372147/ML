#1=Iris-setosa;2=Iris-versicolor;3=Iris-virginica
import random as rd
import math

train_number = 130
test_number = 20


if True:
    f = open('traindata.txt', 'r', encoding='utf-8')
    traindata = f.read()
    f.close()
    traindata = traindata.split('\n')
    for i in range(len(traindata)):
        traindata[i] = traindata[i].split('\t')

    f = open('testdata.txt', 'r', encoding='utf-8')
    testdata = f.read()
    f.close()
    testdata = testdata.split('\n')
    for i in range(len(testdata)):
        testdata[i] = testdata[i].split('\t')


def equal(i, j):
    if i == j:
        return 1
    else:
        return 0


class Neuron:
    weight = []
    out = 0
    threshold = 0

    def cal_in(self, *a):
        for i in a:
            self.out += i
            self.out -= self.threshold
        return self.out

    def cal_other(self, *a):
        for i in a:
            self.out += i
            self.out -= self.threshold
            self.out = 1/(1+math.exp(-self.out))
        return self.out

    def __init__(self, n):
        for i in range(n):
            self.weight.append(rd.random())
        self.threshold = rd.random()


class NN:
    in_number = 4
    hide_number = 4#可调
    out_number = 3
    L_in = []
    L_hide = []
    L_out = []
    error = 999
    lasterror = 999
    eta1=0.05#可调
    eta2=0.05#可调
    min_error = 0.1#可调
    time = 1

    def create(self):
        for i in range(self.in_number):
            self.L_in.append(Neuron(self.hide_number))
            self.L_in[i].threshold = 0
        for i in range(self.hide_number):
            self.L_hide.append(Neuron(self.out_number))
        for i in range(self.out_number):
            self.L_out.append(Neuron(0))

    def calculate(self):
        while self.min_error < self.error and self.time < 10000:
            if self.error > self.lasterror:
                print('误差上升 exit')
                break
            self.lasterror=self.error
            self.time += 1
            self.error = 0
            for j in range(train_number):#遍历一遍traindata 每进行一个样本 改一次权值 标准bp算法

                for i in range(self.in_number):
                    self.L_in[i].out = 0
                    self.L_in[i].cal_in(float(traindata[j][i]))

                for i in range(self.hide_number):
                    temp = 0
                    self.L_hide[i].out = 0
                    for k in range(self.in_number):
                        temp += self.L_in[k].out * self.L_in[k].weight[i]
                    self.L_hide[i].cal_other(temp)

                for i in range(self.out_number):
                    temp = 0
                    self.L_out[i].out = 0
                    for k in range(self.hide_number):
                        temp += self.L_hide[k].out * self.L_hide[k].weight[i]
                    self.L_out[i].cal_other(temp)

                #更改权值阈值
                g = []
                for i in range(self.out_number):
                    g.append(self.L_out[i].out*(1-self.L_out[i].out)*(equal(i+1, int(traindata[j][4]))-self.L_out[i].out))

                e = []
                for h in range(self.hide_number):
                    temp = 0
                    for i in range(self.out_number):
                        temp += g[i]*self.L_hide[h].weight[i]
                    e.append(temp*self.L_hide[h].out*(1-self.L_hide[h].out))

                for i in range(self.in_number):
                    for h in range(self.hide_number):
                        self.L_in[i].weight[h] += float(traindata[j][i])*self.eta2*e[h]

                for h in range(self.hide_number):
                    self.L_hide[h].threshold -= self.eta2*e[h]

                for h in range(self.hide_number):
                    for i in range(self.out_number):
                        self.L_hide[h].weight[i] += self.eta1*g[i]*self.L_hide[h].out

                for i in range(self.out_number):
                    self.L_out[i].threshold -= self.eta1*g[i]

            for j in range(train_number):#新权值算一遍累积误差

                for i in range(self.in_number):
                    self.L_in[i].out = 0
                    self.L_in[i].cal_in(float(traindata[j][i]))

                for i in range(self.hide_number):
                    temp = 0
                    self.L_hide[i].out = 0
                    for k in range(self.in_number):
                        temp += self.L_in[k].out * self.L_in[k].weight[i]
                    self.L_hide[i].cal_other(temp)

                for i in range(self.out_number):
                    temp = 0
                    self.L_out[i].out = 0
                    for k in range(self.hide_number):
                        temp += self.L_hide[k].out * self.L_hide[k].weight[i]
                    self.L_out[i].cal_other(temp)
                    #print('%.2f'%self.L_out[i].out, end=' ')

                temperror = 0
                for i in range(self.out_number):
                    temperror += 0.5*(self.L_out[i].out-equal(i+1, int(traindata[j][4])))**2
                self.error += temperror
            self.error /= train_number
            print(self.time,'误差:',self.error)

    def accuracy_train(self):
        ac = 0
        for j in range(train_number):  #训练集正确率 正确个数/train_number

            for i in range(self.in_number):
                self.L_in[i].out = 0
                self.L_in[i].cal_in(float(traindata[j][i]))

            for i in range(self.hide_number):
                temp = 0
                self.L_hide[i].out = 0
                for k in range(self.in_number):
                    temp += self.L_in[k].out * self.L_in[k].weight[i]
                self.L_hide[i].cal_other(temp)

            for i in range(self.out_number):
                temp = 0
                self.L_out[i].out = 0
                for k in range(self.hide_number):
                    temp += self.L_hide[k].out * self.L_hide[k].weight[i]
                self.L_out[i].cal_other(temp)

            if self.L_out[0].out>self.L_out[1].out and self.L_out[0].out>self.L_out[2].out and traindata[j][4]=='1':
                ac += 1
            else:
                if self.L_out[1].out>self.L_out[0].out and self.L_out[1].out>self.L_out[2].out and traindata[j][4]=='2':
                    ac += 1
                else:
                    if self.L_out[2].out>self.L_out[0].out and self.L_out[2].out>self.L_out[1].out and traindata[j][4]=='3':
                        ac += 1

        print('训练集正确率:',ac/train_number)

    def accuracy_test(self):
        self.error = 0
        for j in range(test_number):  # 累积误差

            for i in range(self.in_number):
                self.L_in[i].out = 0
                self.L_in[i].cal_in(float(testdata[j][i]))

            for i in range(self.hide_number):
                temp = 0
                self.L_hide[i].out = 0
                for k in range(self.in_number):
                    temp += self.L_in[k].out * self.L_in[k].weight[i]
                self.L_hide[i].cal_other(temp)

            for i in range(self.out_number):
                temp = 0
                self.L_out[i].out = 0
                for k in range(self.hide_number):
                    temp += self.L_hide[k].out * self.L_hide[k].weight[i]
                self.L_out[i].cal_other(temp)

            temperror = 0
            for i in range(self.out_number):
                temperror += 0.5 * (self.L_out[i].out - equal(i+1, int(testdata[j][4]))) ** 2
            self.error += temperror
        self.error /= test_number
        print('测试集累积误差:', self.error)
        ac = 0
        for j in range(test_number):  # 测试集正确率

            for i in range(self.in_number):
                self.L_in[i].out = 0
                self.L_in[i].cal_in(float(testdata[j][i]))

            for i in range(self.hide_number):
                temp = 0
                self.L_hide[i].out = 0
                for k in range(self.in_number):
                    temp += self.L_in[k].out * self.L_in[k].weight[i]
                self.L_hide[i].cal_other(temp)

            for i in range(self.out_number):
                temp = 0
                self.L_out[i].out = 0
                for k in range(self.hide_number):
                    temp += self.L_hide[k].out * self.L_hide[k].weight[i]
                self.L_out[i].cal_other(temp)

            if self.L_out[0].out > self.L_out[1].out and self.L_out[0].out > self.L_out[2].out and testdata[j][4] == '1':
                ac += 1
            else:
                if self.L_out[1].out > self.L_out[0].out and self.L_out[1].out > self.L_out[2].out and testdata[j][4] == '2':
                    ac += 1
                else:
                    if self.L_out[2].out > self.L_out[0].out and self.L_out[2].out > self.L_out[1].out and testdata[j][4] == '3':
                        ac += 1

        print('测试集正确率:', ac / test_number)


if __name__ == '__main__':
    nn = NN()
    nn.create()

    nn.accuracy_train()
    nn.accuracy_test()

    nn.calculate()

    nn.accuracy_train()
    nn.accuracy_test()
