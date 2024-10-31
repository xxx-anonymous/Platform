import numpy as np
import random
import pickle
import tensorflow as tf
from keras.utils.np_utils import to_categorical

# x_train: (77400, 32, 32, 3)
# y_train: (77400, 43)
# x_test: (13459, 32, 32, 3)
# y_test: (13459, 43)
# 2113 = 1800 + 313
x = np.load('x.npy')
y = np.load('y.npy')
rand = random.sample(range(2113), 2113)
x_train = tf.gather(x[0:2113], rand[0:1800])
y_train = tf.gather(y[0:2113], rand[0:1800])
x_test = tf.gather(x[0:2113], rand[1800:2113])
y_test = tf.gather(y[0:2113], rand[1800:2113])
for i in range(43):
    if i!=0:
        x_train = tf.concat([x_train, tf.gather(x[i*2113:(i+1)*2113], rand[0:1800])], axis=0)
        y_train = tf.concat([y_train, tf.gather(y[i * 2113:(i + 1) * 2113], rand[0:1800])], axis=0)
        x_test = tf.concat([x_test, tf.gather(x[i * 2113:(i + 1) * 2113], rand[1800:2113])], axis=0)
        y_test = tf.concat([y_test, tf.gather(y[i * 2113:(i + 1) * 2113], rand[1800:2113])], axis=0)

print(tf.shape(x_train))
print(tf.shape(y_train))
print(tf.shape(x_test))
print(tf.shape(y_test))

t = []
for k in range(43):
    t.append(0)
for j in y_test:
    a = np.argmax(j)
    t[a] += 1
print(t)