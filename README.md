Tensorflow从零开始


2020.8.26 前向传播（张量）测试：数据集：MNIST，epoch：10，loss：0.083，lr：1e-3
疑惑点：梯度爆炸相关知识


2020.8.27 数据集的相关操作，
合并与切割:tf.concat,split,stack,unstack
数据统计：tf.norm,reduce_max/min,argmax/min,equal,unique
张量排序：tf.sort/argsort,Topk,Top-5 Acc
填充与复制：tf.pad，tile，broadcast_to
张量限幅：tf.clip_by_value,relu(max(x,0)),clip_by_norm,clip_by_global_norm(gradient clipping)