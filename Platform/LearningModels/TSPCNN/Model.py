import numpy as np  # linear algebra
import pickle
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras import optimizers
import random
from keras.datasets import cifar10
from keras.datasets import mnist


# x_train: (86989, 32, 32, 3)
# y_train: (86989, 43)
# x_test: (12630, 32, 32, 3)
# y_test: (12630,)
# x_validation: (4410, 32, 32, 3)
# y_validation: (4410, 43)
# labels: 43
class CNN:
    def __init__(self):
        self.model = self.build()
        self.model.compile(optimizer=optimizers.Adam(0.005), loss='categorical_crossentropy', metrics=['accuracy'])
        self.epochs = 1
        #self.annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (0.5 * x))
        #self.x_train, self.y_train, self.x_validation, self.y_validation, self.x_test, self.y_test = self.gen_data()

    @staticmethod
    def build():

        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(43, activation='softmax'))

        # model = Sequential()
        # model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
        # model.add(MaxPool2D(pool_size=2))
        # model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        # model.add(MaxPool2D(pool_size=2))
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(10, activation='softmax'))
        '''
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPool2D(pool_size=2))
        model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(84, activation='relu'))
        model.add(Dense(43, activation='softmax'))
        '''
        return model

    def train(self, x_train, y_train, x_validation, y_validation):
        self.model.fit(x_train, y_train,
                       batch_size=100, epochs=self.epochs,
                       #validation_data=(x_validation, y_validation),
                       #callbacks=[self.annealer],
                       verbose=1)

    def evaluate(self, x_test, y_test):
        loss, acc = self.model.evaluate(x_test, y_test, batch_size=100)
        return loss, acc
    '''
    def gen_data(self):
        with open('TSP-data/data2.pickle', 'rb') as f:
            data = pickle.load(f, encoding='latin1')  # dictionary type

        # Preparing y_train and y_validation for using in Keras
        data['y_train'] = to_categorical(data['y_train'], num_classes=43)
        data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)

        # Making channels come at the end
        data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
        data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
        data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)
        return  data['x_train'], data['y_train'], data['x_validation'], data['y_validation'], data['x_test'], data['y_test']
    '''
    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()



if __name__ == '__main__':
    cnn = CNN()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    '''
    for i in range(10):
        cnn.train(x_train, y_train, x_train, y_train)
        loss, acc = cnn.evaluate(x_test, y_test)
        print(acc)
        '''
    print('ok')
    w = np.load('site.npy', allow_pickle=True)
    print('ok')
    cnn.model.set_weights(w)
    loss, acc = cnn.evaluate(x_test, y_test)
    print(acc)