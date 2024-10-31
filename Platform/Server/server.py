import random
import socket
import os
import threading
from Server.handle import Handle
import queue
import numpy as np
from Transaction.Transaction import transaction
from LearningModels.TSPCNN.Model import CNN
import time
import pickle
from keras.utils.np_utils import to_categorical
import zmq

# TCP编码
code = 'gbk'


class TCPServer(threading.Thread):
    def __init__(self, host, port, name):
        super().__init__()
        self.name = name
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((host, port))
        print('服务端开启')
        self.s.listen(128)
        print('监听开启')
        self.mutex = threading.Lock()  # 线程锁
        self.cnn = CNN()
        self.trans_pool = {}  # Tips, name:trans
        self.iterations = [0, ]  # 当前迭代，使用len()获取

        # self.client_index = [0] * 100  # 100个的客户端接入状态，接入则标志置为1
        self.client_trans_num = [0] * 100  # 100个的客户端的交易数量
        self.client_trans_approved_num = [0] * 100  # 100个的客户端的交易被直接批准的次数

        context = zmq.Context()
        self.socket_web = context.socket(zmq.PUB)
        self.socket_web.bind("tcp://*:5411")

        self.x_test, self.y_test = self.init()
        # wandb.init(project='dsl')

    def init(self):
        # 创建并添加一个初始交易
        trans = transaction(1, 0, time.time(), self.cnn.get_weights(), [])
        self.trans_pool[trans.name] = trans
        init_trans = {'type': 1,
                      'ID': trans.name,
                      'Trans_hash': trans.hash[:5] + '...' + trans.hash[-6:-1],
                      'previous': [],
                      'trans_num': 0,
                      'weight': -1,
                      'Accuracy': 0,
                      'located': [0, 3]}
        time.sleep(0.5)
        self.socket_web.send_pyobj(init_trans)

        # 生成测试集
        with open('../dataset/TSP-data/data2.pickle', 'rb') as f:
            data = pickle.load(f, encoding='latin1')  # dictionary type
        data['y_test'] = to_categorical(data['y_test'], num_classes=43)
        data['x_test'] = data['x_test'].transpose(0, 2, 3, 1)

        return data['x_test'], data['y_test']

        # x_test = np.load('../dataset/CIFAR10/cifar10-x_text.npy')
        # y_test = np.load('../dataset/CIFAR10/cifar10-y_test.npy')

        # x_test = np.load('../dataset/MNIST/x_test.npy')
        # y_test = np.load('../dataset/MNIST/y_test.npy')
        # return x_test, y_test

    def run(self):
        # 客户端连接记录
        q = queue.Queue()
        record_connect = threading.Thread(target=self.record_connect, args=(q,))
        record_connect.start()

        # 更新交易池（Tips）
        updata_record_trans = threading.Thread(target=self.update_trans_pool)
        updata_record_trans.start()

        # 测试当前全局模型精度 and 向web添加新交易
        new_trans = queue.Queue()
        test_acc = threading.Thread(target=self.test_now_acc, args=(20, new_trans))
        test_acc.start()

        while True:
            print('等待连接...')
            client, address = self.s.accept()

            # 创建一个新的线程为刚刚到来的客户服务
            handle_threading = Handle(client, address, q, new_trans, self.trans_pool, self.iterations, self.client_trans_num, self.client_trans_approved_num)
            handle_threading.start()

    def __del__(self):
        self.s.close()
        # wandb.finish()

    def record_connect(self, q):
        while True:
            record = q.get()
            with open('connectRecord.txt', 'a') as f:
                f.write(record)

    def update_trans_pool(self):
        while True:
            del_lst = []  # 待删除列表
            self.mutex.acquire()  # 上锁
            length = len(self.iterations)
            for name, trans in self.trans_pool.items():
                if trans.approved > 0 or (length - trans.iterations) > 30:
                    del_lst.append(name)
            self.mutex.release()  # 解锁
            if len(del_lst) > 0:
                # with open('transactionsRecord.txt', 'a') as f:
                for del_trans in del_lst:
                    # content = self.get_trans_content(self.trans_pool[del_trans])
                    # f.write(content)
                    del self.trans_pool[del_trans]
            time.sleep(0.5)

    def get_trans_content(self, trans):
        name = str(trans.name)
        approved_num = str(trans.approved)
        approval = str(trans.approval)
        acc = str(trans.acc_choice)

        content = name + ": " + "acc = {}, ".format(acc) + "approved_num [{}] , ".format(
            approved_num) + "approval_two_tips {}\n".format(approval + "iterations {}".format(trans.iterations))
        return content

    def test_now_acc(self, interval, new_trans):
        num = interval
        while True:
            # 当前模型精度
            if num <= len(self.iterations):
                trans_pool = self.trans_pool.copy()
                max_acc = 0
                for name, trans in trans_pool.items():
                    self.cnn.set_weights(trans.model)
                    loss, acc = self.cnn.evaluate(self.x_test, self.y_test)
                    if max_acc < acc:
                        max_acc = acc
                content = 'iterations: {}, acc: {}, tips_num: {}\n'.format(num, max_acc, len(trans_pool))
                with open('accRecord{}.txt'.format(self.name), 'a') as f:
                    f.write(content)
                with open('clientTransRecord{}.txt'.format(self.name), 'a') as f:
                    f.write(str(self.client_trans_num) + '\n' + str(self.client_trans_approved_num) + '\n')
                #wandb.log({'acc': max_acc, 'Tips_num': len(trans_pool)})
                msg = {'type': 0,
                       'iterations': num,
                       'Accuracy': max_acc,
                       'Tips_num': len(trans_pool),
                       'Client_num': sum(1 for x in self.client_trans_num if x != 0),
                       'Client_trans_num': self.client_trans_num,
                       'Client_approved_num': self.client_trans_approved_num}
                self.socket_web.send_pyobj(msg)
                num += interval

            # 向web添加新交易
            trans_temp = new_trans.get()
            trans = {'type': 1,
                     'ID': trans_temp.name,
                     'Trans_hash': trans_temp.hash[:5] + '...' + trans_temp.hash[-5:-1],
                     'previous': trans_temp.approval,
                     'trans_num': trans_temp.iterations,
                     'weight': -1,
                     'Accuracy': trans_temp.acc_choice,
                     'located': []}
            if trans['trans_num'] % 3 == 0:
                trans['located'] = [trans['trans_num'], np.random.uniform(1, 3)]
            elif trans['trans_num'] % 3 == 1:
                trans['located'] = [trans['trans_num'], np.random.uniform(6, 9)]
            else:
                trans['located'] = [trans['trans_num'], np.random.uniform(3, 6)]
            try:
                self.socket_web.send_pyobj(trans)
            except:
                pass
