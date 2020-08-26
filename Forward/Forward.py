# Author: Betterman
# -*- coding = utf-8 -*-
# @Time : 2020/8/26 16:43
# @File : Forward.py
# @Software : PyCharm
import os
#CPP打印信息，0：全打印 1：打印warning 2：打印error 3：打印FATAL
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

#加载MNIST数据集
#x:[60k,28,28] y:[60k,]
(x,y),_ = datasets.mnist.load_data()

#转换为Tensor
#x:[0-255]=>[0-1.]
x = tf.convert_to_tensor(x,dtype=tf.float32) /255
y = tf.convert_to_tensor(y,dtype=tf.int32)

print(x.shape,x.dtype,y.shape,y.dtype)
print(tf.reduce_min(x),tf.reduce_max(x),tf.reduce_min(y),tf.reduce_max(y))

#创建数据集
#将x_train和y_train from_tensor_slices函数切分传入的 Tensor 的第一个维度，生成相应的 dataset
#关联x_train和y_train关联后为Dataset，无法正常输出【可以理解为很多块batch】
train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
#通过迭代器输出
train_iter = iter(train_db)
sample = next(train_iter)
#还可以通过 for step, (x_train, y_train) in enumerate(train_db):此时x_train和y_train都为张量
print('batch：',sample[0].shape,sample[1].shape)



#创建权值 w b
#[b,784] => [b,256] => [b,128] => [b,10]
#tf.random.truncated_normal: 截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。
#默认均值mean = 0，方差 stddev = 1，会发生梯度爆炸，故修改
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1,seed=1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1,seed=2))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1,seed=3))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3
#迭代次数
epoch = 10
for epoch in range(epoch):
    for step,(x,y) in enumerate(train_db): #对于每个batch
    #x:[128,28,28] y:[128]
    #变换shape
        x = tf.reshape(x,[-1,28*28])

        with tf.GradientTape() as tape:
        # h1 = x @ w1 + b1
        #已经自动broadcas_to转换
            h1 = x@w1 + b1
        #relu非线性处理
            h1 = tf.nn.relu(h1)

            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)

            out = h2@w3 + b3
        #计算loss
        #out:[b,10] y = [10]
            y_onehot = tf.one_hot(y,depth=10)
        #mse = mean（sum(y-out)**2）
            loss = tf.square(y_onehot - out)
        #mean:scalar
            loss = tf.reduce_mean(loss)

    #计算梯度
    #此处grads=[dw1,db1,dw2,db2,dw3,db3]
        grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
    #如果此时grads类型是None，说明前面W和B没有扩展成tf.Variable || tf.Constant
    #更新梯度
    #如果这里报错unsupported operand type(s) for *: 'float' and 'NoneType'
    #说明相减后生成的不是Variable而是tensor，所以要原地更新assign_sub()
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
        w2.assign_sub(lr*grads[2])
        b2.assign_sub(lr*grads[3])
        w3.assign_sub(lr*grads[4])
        b3.assign_sub(lr*grads[5])

    #如果显示nan，说明出现了梯度爆炸现象
    #爆炸解决方法1：去修改初始化权值的范围
        if step % 100 == 0:
            print("epoch:",epoch,step,"Loss:",float(loss))

