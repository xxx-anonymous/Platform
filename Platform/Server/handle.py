import socket
import os
import threading
import queue
import time
from Transaction.Transaction import transaction
from LearningModels.TSPCNN.Model import CNN
import numpy as np
import pickle
from keras.utils.np_utils import to_categorical
import random
import tqdm


# TCP编码
code = 'gbk'

# TCP通信缓存
BUFFER_SIZE = 1024


def generate_val_data_TSP():
    with open('../dataset/TSP-data/data2.pickle', 'rb') as f:
        data = pickle.load(f, encoding='latin1')  # dictionary type
    data['y_test'] = to_categorical(data['y_test'], num_classes=43)
    data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

    ran = random.sample(range(12630), 3000)
    choice_x = data['x_test'][ran]
    choice_y = data['y_test'][ran]

    return choice_x, choice_y

def generate_val_data_CIFAR10():
    x_test = np.load('../dataset/CIFAR10/cifar10-x_text.npy')
    y_test = np.load('../dataset/CIFAR10/cifar10-y_test.npy')

    ran = random.sample(range(10000), 2000)
    choice_x = x_test[ran]
    choice_y = y_test[ran]

    return choice_x, choice_y

def generate_val_data_MNIST():
    x_test = np.load('../dataset/MNIST/x_test.npy')
    y_test = np.load('../dataset/MNIST/y_test.npy')

    ran = random.sample(range(10000), 2000)
    choice_x = x_test[ran]
    choice_y = y_test[ran]

    return choice_x, choice_y



class Handle(threading.Thread):
    def __init__(self, client_socket, client_address, q, new_trans_q, trans_pool, iterations, client_trans_num, client_trans_approved_num):
        super().__init__()
        self.client_socket = client_socket
        self.client_address = client_address
        self.q = q
        self.new_trans_q = new_trans_q
        self.mutex = threading.Lock()
        self.trans_pool = trans_pool  # 指向server.trans_pool字典的地址
        self.iterations = iterations  # 指向server.iterations列表的地址
        self.client_trans_num = client_trans_num
        self.client_trans_approved_num = client_trans_approved_num
        self.x_val, self.y_val = generate_val_data_TSP()

    # def init(self):
    #     with open('../dataset/TSP-data/data2.pickle', 'rb') as f:
    #         data = pickle.load(f, encoding='latin1')  # dictionary type
    #     data['y_test'] = to_categorical(data['y_test'], num_classes=43)
    #     data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)
    #
    #     ran = random.sample(range(12630), 3000)
    #     choice_x = data['x_test'][ran]
    #     choice_y = data['y_test'][ran]
    #
    #     return choice_x, choice_y

    def run(self):
        # print('IP is: {}'.format(self.client_address[0]))
        # print('Port is: {}'.format(self.client_address[1]))
        record = '---------新客户端连接---------\n' + '-IP is: {}\n'.format(
            self.client_address[0]) + '-Port is: {}\n'.format(self.client_address[1]) + '-Time: {}\n'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())) + '----------------------------\n'
        self.q.put(record)


        while True:
            try:
                req = self.client_socket.recv(BUFFER_SIZE).decode(code)
                if req:
                    if req == 'Downloading':
                        #print('Receive Downloading req')
                        self.send_tips()
                    elif req == 'Uploading':
                        print('Receive Uploading req')
                        self.receive_newSite()
                else:
                    break
            except:
                pass
        self.client_socket.close()

    def send_tips(self):
        trans_list = self.select_some_tips(5)
        data = pickle.dumps(trans_list)
        size = len(data)
        self.client_socket.sendall(str(size).encode(code))  # 发送数据大小
        self.client_socket.recv(BUFFER_SIZE)  # 等待应答

        for i in range(0, len(data), BUFFER_SIZE):
            try:
                self.client_socket.sendall(data[i:i + BUFFER_SIZE])
            except:
                pass

    def receive_newSite(self):
        size = int(self.client_socket.recv(BUFFER_SIZE).decode(code))  # 接收带接收数据的大小
        self.client_socket.sendall('ok'.encode(code))  # 返回应答

        #progress = tqdm.tqdm(range(size), '接收', unit='B', unit_divisor=1024, unit_scale=True)
        data = b''
        recv_size = 0
        while recv_size < size:
            recv = self.client_socket.recv(BUFFER_SIZE)
            data += recv
            recv_size += len(recv)
            #progress.update(len(recv))
        trans = pickle.loads(data)
        trans = self.package_new_trans(trans)
        #self.mutex.acquire()  # 上锁
        self.add_new_trans(trans)
        #self.mutex.release()  # 解锁

        self.client_trans_num[int(trans.publisher)] += 1
        for t in trans.approval:
            self.client_trans_approved_num[int(t.split('_')[0])] += 1

    def package_new_trans(self, trans):
        cnn = CNN()
        cnn.set_weights(trans.model)
        loss, acc = cnn.evaluate(self.x_val, self.y_val)
        trans.acc_choice = acc  # 添加测试精度，作为Tips选择时的参考
        # trans.timespan = time.strftime("%y-%m-%d-%H:%M-:%S", time.gmtime())
        trans.timespan = time.time()  # 将交易的时间改为服务端上链时间
        trans.name = trans.gen_name()  # 更新交易名称
        trans.iterations = len(self.iterations)  # 添加该交易上传时的全局迭代次数
        return trans

    def select_some_tips(self, k):
        trans_pool = self.trans_pool.copy()
        visible_trans = []  # 可选择tips
        for name, trans in trans_pool.items():
            visible_trans.append(trans)
        if len(visible_trans) == 0:
            print("Terrible Error: Have none transaction to select")

        if len(visible_trans) <= k:
            return visible_trans
        else:
            rand_seq = random.sample(range(len(visible_trans)), k)
            right_lst = []
            for seq in rand_seq:
                right_lst.append(visible_trans[seq])
            return right_lst

    def add_new_trans(self, trans):
        for name in trans.approval:
            # 更新其指向的交易信息
            try:
                self.trans_pool[name].approved += 1
                self.trans_pool[name].approved_time.append(trans.timespan)
            except:
                pass
        self.trans_pool[trans.name] = trans
        self.iterations.append(0)  # 全局迭代次数+1
        self.new_trans_q.put(trans)
