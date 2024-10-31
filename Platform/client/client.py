import socket
import os
import pickle
import threading
from Transaction.Transaction import transaction
from LearningModels.TSPCNN.Model import CNN
import tqdm
import math
from keras.utils.np_utils import to_categorical
import numpy as np
import time
import random

# 服务器IP和端口号
host = '192.168.31.10'
port = 8800

# TCP编码
code = 'gbk'

# TCP通信缓存
BUFFER_SIZE = 1024


class Client:
    def __init__(self, client_name):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.name = client_name
        print('客户端{}开启'.format(self.name))

        self.x_train, self.y_train, self.x_val, self.y_val = self.CNN_data_separate()

        self.connect_2_server()

    def connect_2_server(self):
        try:
            self.s.connect((host, port))
            print('已连接到服务器')
        except:
            print('连接失败')

    def send_req(self, req):
        self.s.send(req.encode(code))

    def recv_tips(self):
        size = int(self.s.recv(BUFFER_SIZE).decode(code))  # 接收待接收数据的大小
        self.s.send('ok'.encode(code))  # 返回应答

        progress = tqdm.tqdm(range(size), '客户端{}接收'.format(self.name), unit='B', unit_divisor=1024, unit_scale=True)  # 创建接收进度条
        data = b''
        recv_size = 0
        while recv_size < size:
            recv = self.s.recv(BUFFER_SIZE)
            data += recv
            recv_size += len(recv)
            progress.update(len(recv))
        trans_list = pickle.loads(data)
        return trans_list  # 返回接收到的Tips列表

    def select_tips(self, trans_list, k):
        if len(trans_list) <= k:
            choose_trans_lst = trans_list
        else:
            acc = []
            for trans in trans_list:
                acc.append(trans.acc_choice)
            temp = []
            Inf = -1
            for i in range(k):
                temp.append(acc.index(max(acc)))
                acc[acc.index(max(acc))] = Inf
            choose_trans_lst = []
            for i in temp:
                choose_trans_lst.append(trans_list[i])
        return choose_trans_lst

    def aggregation(self, some_tips):
        weight_lst = []
        for trans in some_tips:
            weight_lst.append(trans.model)
        weight_lst = np.array(weight_lst)

        lenth = len(some_tips)
        global_weight = np.sum(weight_lst, axis=0) / lenth
        return global_weight

    def local_train(self, weight):
        cnn = CNN()
        cnn.set_weights(weight)
        cnn.train(self.x_train, self.y_train, self.x_val, self.y_val)
        w = cnn.get_weights()
        return w

    def generate_site(self, new_weight, choice_tips):
        approv_lst = []
        for t in choice_tips:
            approv_lst.append(t.name)
        new_trans = transaction(1, self.name, time.time(), new_weight, approv_lst)
        return new_trans

    def send_newSite(self, new_site):
        data = pickle.dumps(new_site)
        size = len(data)
        self.s.send(str(size).encode(code))  # 发送数据大小
        self.s.recv(BUFFER_SIZE)  # 接收应答
        for i in range(0, size, BUFFER_SIZE):
            self.s.sendall(data[i: i + BUFFER_SIZE])

    def run_one_round(self):
        self.send_req('Downloading')
        trans_list = self.recv_tips()
        choice_two_tips = self.select_tips(trans_list, 2)
        weight = self.aggregation(choice_two_tips)
        new_weight = self.local_train(weight)
        new_site = self.generate_site(new_weight, choice_two_tips)
        self.send_req('Uploading')
        self.send_newSite(new_site)

    def CNN_data_separate(self):
        with open('../dataset/TSP-data/data2.pickle', 'rb') as f:
            data = pickle.load(f, encoding='latin1')  # dictionary type

        # Preparing y_train and y_validation for using in Keras
        data['y_train'] = to_categorical(data['y_train'], num_classes=43)
        data['y_validation'] = to_categorical(data['y_validation'], num_classes=43)
        data['y_test'] = to_categorical(data['y_test'], num_classes=43)

        # Making channels come at the end
        data['x_train'] = data['x_train'].transpose(0, 2, 3, 1)
        data['x_validation'] = data['x_validation'].transpose(0, 2, 3, 1)
        data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

        rand = random.sample(range(86968), 2000)

        return data['x_train'][rand], data['y_train'][rand], data['x_validation'], data['y_validation']

    def __del__(self):
        self.s.close()
