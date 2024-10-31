import socket
import os
from client.client import Client
import time
import threading

def run(i):
    client = Client(i)
    client.run_one_round()
    del client
if __name__ == '__main__':
    while True:
        for i in range(5):
            while len(threading.enumerate()) >= 3:
                pass
            client_run = threading.Thread(target=run, args=(i+16,))
            client_run.start()
            time.sleep(4)
            print(len(threading.enumerate()))
