# Author: Betterman
# -*- coding = utf-8 -*-
# @Time : 2020/8/27 14:56
# @File : acc_topK.py
# @Software : PyCharm
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import  tensorflow as tf

tf.random.set_seed(2467)
#计算accuracy
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]
    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    # [10, b]
    correct = tf.equal(pred, target_)
    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k* (100.0 / batch_size) )
        res.append(acc)
    return res
#正态分布10个样本，6个类
output = tf.random.normal([10, 6])
#softmax使得6类总和概率为1
output = tf.math.softmax(output, axis=1)
#maxval =6从0-5中随机生成10个label
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
print('prob:', output.numpy())
pred = tf.argmax(output, axis=1)
print('pred:', pred.numpy())
print('label:', target.numpy())
acc = accuracy(output, target, topk=(1,2,3,4,5,6))
print('top-1-6 acc:', acc)