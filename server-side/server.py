import select
import socket

import time
from threading import Thread
import threading

def send(connection):

    response_body = "content".encode("utf-8")
    response_header = "HTTP/1.1 200 OK\r\n"
    response_header += "Content-Length: %d\r\n" % len(response_body)
    response_header += "\r\n"
    for i in range(25):
        thread_name = threading.current_thread().name
        time.sleep(1)
        print(thread_name)
    connection.send(response_header.encode("utf-8") + response_body)
    
    print("发送完成", time.time())


response = b''  
  
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
serversocket.bind(('127.0.0.1', 8848))  
serversocket.listen(1)  
# 因为socket默认是阻塞的，所以需要使用非阻塞（异步）模式。  
serversocket.setblocking(0)  
  
# 创建一个epoll对象  
epoll = select.epoll()
epoll.register(serversocket.fileno(), select.EPOLLIN)  
  
try:  
    connections = {}
    count = 0
    while True:
        print("==================================================================================")
        events = epoll.poll(1)
        for fileno, event in events: 
            if fileno == serversocket.fileno():  
                connection, address = serversocket.accept()  
                print('client connected:', address)  
                connection.setblocking(0)  
                epoll.register(connection.fileno(), select.EPOLLIN)  
                connections[connection.fileno()] = connection  
            elif event & select.EPOLLIN:  
                count += 1
                print("------recv---------", count)
                request = b''
                while True:
                    # 将收到的数据拼接起来
                    temp = connections[fileno].recv(1024)
                    request += temp
                    if len(temp) < 1024:
                        break

                if request:
                    epoll.modify(fileno, 0)
                    print("================================")

                    response_body = "content".encode("utf-8")
                    response_header = "HTTP/1.1 200 OK\r\n"
                    response_header += "Content-Length: %d\r\n" % len(response_body)
                    response_header += "\r\n"
                    for i in range(25):
                        time.sleep(1)
                        print("-------------------------------------")
                    connections[fileno].send(response_header.encode("utf-8") + response_body)

                    # epoll.modify(fileno, select.EPOLLIN)
                    # t = Thread(target=send, args=(connections[fileno],))
                    # t.start()
                else:
                    epoll.unregister(fileno)  
                    connections[fileno].close()  
                    del connections[fileno]  
                    print("断开连接")

            elif event & select.EPOLLOUT:  

                print("-------send---------")
                # response_body = "content".encode("utf-8")
                # response_header = "HTTP/1.1 200 OK\r\n"
                # response_header += "Content-Length: %d\r\n" % len(response_body)
                # response_header += "\r\n"
                # connections[fileno].send(response_header.encode("utf-8") + response_body)
                # epoll.modify(fileno, select.EPOLLIN) 

                # print("================================")
                # epoll.modify(fileno, select.EPOLLIN)
                # t = Thread(target=send, args=(connections[fileno],))
                # t.start()
                # t.join()
                

finally:  
    epoll.unregister(serversocket.fileno())  
    epoll.close()  
    serversocket.close()
