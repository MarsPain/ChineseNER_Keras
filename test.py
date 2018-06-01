# test3
# import tensorflow as tf
#
# a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
# b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
# c = a + b

# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# sess.run(tf.global_variables_initializer())
# print(sess.run(c))

# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#矩阵点乘
# # l5 = [[0],[0],[1]]
# # l5 = [0]
# # l5 = [[0], [1]]
# l5 = [[0],[1],[2]]
# l5 = np.asarray(l5)
# print(l5.shape)
# # l6 = [[0, 0, 1]]
# # l6 =  [[0,0,1], [0,0,1], [0,0,1]]
# l6 =  [[[0,0,1]], [[0,0,2]], [[0,0,3]]]
# l6 = np.asarray(l6)
# print(l6.shape)
# l7 = l5 * l6    #l5中所有元素依次与l6中每个元素进行点乘，然后依次得到结果矩阵的子矩阵
# print(l7, l7.shape)
#利用矩阵点乘获取标签的转移矩阵
import numpy as np
l = [[[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]],
     [[0, 0, 2], [0, 0, 2], [0, 2, 0], [0, 2, 0]],
     [[0, 0, 3], [0, 0, 3], [0, 3, 0], [0, 3, 0]]]
l = np.asarray(l)
# print("l", l, l.shape)
l1 = l[:, :-1]
l1 = np.expand_dims(l1, 3)
print("l1", l1, l1.shape)
l2 = l[:, 1:]
l2 = np.expand_dims(l2, 2)
print("l2", l2, l2.shape)
l3 = l1 * l2
# 点乘后，每个矩阵的1所在位置揭示了由第几个标签（行数）到第几个标签（列数）的转换
print("l3", l3, l3.shape)