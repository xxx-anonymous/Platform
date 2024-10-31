import socket
import os
import threading
from Server.server import TCPServer
import time

# 本机IP和端口号
host = '192.168.31.10'
port = 8800


if __name__ == '__main__':
    server1 = TCPServer(host, port, 1)
    server1.start()



