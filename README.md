# Computer Networks:

## Experiments:

## [EX-1 STUDY OF SOCKET PROGRAMMING WITH CLIENT-SERVER MODEL](https://github.com/doit-1/2/tree/main#ex-1-study-of-socket-programming-with-client-server-model-1)
## [EX-2 IMPLEMENTATION OF STOP AND WAIT PROTOCOL](https://github.com/doit-1/2/tree/main#ex-2-implementation-of-stop-and-wait-protocol-1)
## [EX-3 IMPLEMENTATION OF SLIDING WINDOW PROTOCOL](https://github.com/doit-1/2/tree/main#ex-3-implementation-of-sliding-window-protocol-1)
## [EX-4 IMPLEMENTATION OF ADDRESS RESOLUTION PROTOCOL (ARP)](https://github.com/doit-1/2/tree/main#ex-4-implementation-of-address-resolution-protocol-arp-1)
## [EX-5 IMPLEMENTATION OF REVERSE ADDRESS RESOLUTION PROTOCOL(RARP)](https://github.com/doit-1/2/tree/main#ex-5-implementation-of-reverse-address-resolution-protocolrarp-1)
## [EX-6 IMPLEMENTATION OF PING COMMAND](https://github.com/doit-1/2/tree/main#ex-6-implementation-of-ping-command-1)
## [EX-7 IMPLEMENTATION OF TRACEROUTE COMMAND](https://github.com/doit-1/2/tree/main#ex-7-implementation-of-traceroute-command-1)
## [EX-8 APPLICATION USING TCP SOCKETS - CREATING ECHO CLIENT-SERVER](https://github.com/doit-1/2/tree/main#ex-8-application-using-tcp-sockets---creating-echo-client-server-1)
## [EX-9 APPLICATION USING TCP SOCKETS - CREATING FOR CHAT CLIENT-SERVER](https://github.com/doit-1/2/tree/main#ex-9-application-using-tcp-sockets---creating-for-chat-client-server-1)
## [EX-10 APPLICATION USING TCP SOCKETS - FILE TRANSFER PROGRAM](https://github.com/doit-1/2/tree/main#ex-10-application-using-tcp-sockets---file-transfer-program-1)

# PROGRAMS:

## EX-1 STUDY OF SOCKET PROGRAMMING WITH CLIENT-SERVER MODEL:
### Client:
```
import socket
from datetime import datetime
s=socket.socket()
s.bind(('localhost',8000))
s.listen(5)
c,addr=s.accept()
print("Client Address : ",addr)
now = datetime.now()
c.send(now.strftime("%d/%m/%Y %H:%M:%S").encode())
ack=c.recv(1024).decode()
if ack:
    print(ack)
    c.close()
```
### Server:
```
import socket
s=socket.socket()
s.connect(('localhost',8000))
print(s.getsockname())
print(s.recv(1024).decode())
s.send("acknowledgement recived from the server".encode())
```
### OUTPUT:
### Client:
![image](https://github.com/doit-1/2/assets/136359575/bfa79282-053c-4832-8410-93ac2b631c15)
### Server:
![image](https://github.com/doit-1/2/assets/136359575/ec93d4b9-9869-4ff8-9940-96fed6752f7e)

## EX-2 IMPLEMENTATION OF STOP AND WAIT PROTOCOL
### Client:
```
import socket
s=socket.socket()
s.bind(('localhost',8000))
s.listen(5)
c,addr=s.accept()
while True:
    i=input("Enter a data: ")
    c.send(i.encode())
    ack=c.recv(1024).decode()
    if ack:
        print(ack)
        continue
    else:
        c.close()
        break
```
### Server:
```
import socket
s=socket.socket()
s.connect(('localhost',8000))
while True:
    print(s.recv(1024).decode())
    s.send("Acknowledgement Recived".encode())
```
### OUTPUT :
### Client Side:
![image](https://github.com/doit-1/2/assets/136359575/0ca80597-6a9f-4bb0-8ccd-c2106529e30b)

### Server Side:
![image](https://github.com/doit-1/2/assets/136359575/825e8649-e32e-416a-820c-2dbda44bd5af)

## EX-3 IMPLEMENTATION OF SLIDING WINDOW PROTOCOL

### Client Side:
```
import socket
s=socket.socket()
s.bind(('localhost',8000))
s.listen(5)
c,addr=s.accept()
size=int(input("Enter number of frames to send : "))
l=list(range(size))
s=int(input("Enter Window Size : "))
st=0
i=0
while True:
    while(i<len(l)):
        st+=s
        c.send(str(l[i:st]).encode())
        ack=c.recv(1024).decode()
        if ack:
            print(ack)
            i+=s
```
### Server Side:
```
import socket
s=socket.socket()
s.connect(('localhost',8000))
while True:
    print(s.recv(1024).decode())
    s.send("acknowledgement recived from the server".encode())
```
### OUTPUT :
### Client Side:
![image](https://github.com/doit-1/2/assets/136359575/3b41d12c-e6ef-40d0-8eac-ec6bd15d5214)

### Server Side:
![image](https://github.com/doit-1/2/assets/136359575/f7b30cf4-ab3c-4ee8-9748-e8216c014494)

## EX-4 IMPLEMENTATION OF ADDRESS RESOLUTION PROTOCOL (ARP)
### Client Side:
```
import socket
s=socket.socket()
s.bind(('localhost',8000))
s.listen(5)
c,addr=s.accept()
address={"165.165.80.80":"6A:08:AA:C2","165.165.79.1":"8A:BC:E3:FA"};
while True:
    ip=c.recv(1024).decode()
    try:
        c.send(address[ip].encode())
    except KeyError:
        c.send("Not Found".encode())
```
### Server Side:
```
import socket
s=socket.socket()
s.connect(('localhost',8000))#
while True:
    ip=input("Enter logical Address : ")
    s.send(ip.encode())
    print("MAC Address",s.recv(1024).decode())
```
### OUTPUT :
### Client Side:
![image](https://github.com/doit-1/2/assets/136359575/6ee7fdc2-de31-48d4-b1f0-8803e233d770)

### Server Side:
![image](https://github.com/doit-1/2/assets/136359575/deeb6a76-3c3a-481b-92ff-dd82d063f93f)

## EX-5 IMPLEMENTATION OF REVERSE ADDRESS RESOLUTION PROTOCOL(RARP)
### Client Side:
```
import socket
s=socket.socket()
s.bind(('localhost',8000))
s.listen(5)
c,addr=s.accept()
address={"6A:08:AA:C2":"192.168.1.100","8A:BC:E3:FA":"192.168.1.99"};
while True:
    ip=c.recv(1024).decode()
    try:
        c.send(address[ip].encode())
    except KeyError:
        c.send("Not Found".encode())
```
### Server Side:
```
import socket
s=socket.socket()
s.connect(('localhost',8000))
while True:
    ip=input("Enter MAC Address : ")
    s.send(ip.encode())
    print("Logical Address",s.recv(1024).decode())
```
### OUTPUT :
### Client Side:
![image](https://github.com/doit-1/2/assets/136359575/dcb967da-cf7d-4ea0-8182-54be84f5dd39)

### Server Side:
![image](https://github.com/doit-1/2/assets/136359575/95d91fd2-c25c-4d32-9cc0-fe0030c4b633)


## EX-6 IMPLEMENTATION OF PING COMMAND
### Client Side:
```
import socket
from pythonping import ping
s=socket.socket()
s.bind(('localhost',8000))
s.listen(5)
c,addr=s.accept()
while True:
    hostname=c.recv(1024).decode()
    try:
        c.send(str(ping(hostname, verbose=False)).encode())
    except KeyError:
        c.send("Not Found".encode())
```
### Server Side:
```
import socket
s=socket.socket()
s.connect(('localhost',8000))
while True:
    ip=input("Enter the website you want to ping ")
    s.send(ip.encode())
    print(s.recv(1024).decode())
```
### OUTPUT :
### Client Side:
![image](https://github.com/doit-1/2/assets/136359575/9973fc77-e04f-447a-bd77-9919699caa6b)

### Server Side:
![image](https://github.com/doit-1/2/assets/136359575/82add942-573f-4fd0-a9e0-932f05190641)


## EX-7 IMPLEMENTATION OF TRACEROUTE COMMAND

### PROGRAM :
```
from scapy.all import*
target = ["www.google.com"]
result, unans = traceroute(target,maxttl=32)
print(result,unans)
```
### OUTPUT :
![image](https://github.com/doit-1/2/assets/136359575/5da94979-f6af-4f04-b89a-4426dd96d80e)

## EX-8 APPLICATION USING TCP SOCKETS - CREATING ECHO CLIENT-SERVER
### Client_Side:
```
import socket
s=socket.socket()
s.connect(('localhost',8000))
while True:
    msg=input("Client > ")
    s.send(msg.encode())
    print("Server > ",s.recv(1024).decode())
```
### Server_Side:
```
import socket
s=socket.socket()
s.bind(('localhost',8000))
s.listen(5)
c,addr=s.accept()
while True:
    ClientMessage=c.recv(1024).decode()
    c.send(ClientMessage.encode())
```
### OUTPUT :
### Client_Side:
![image](https://github.com/doit-1/2/assets/136359575/7b5ca8ab-a154-43ab-a8cb-f01be584920e)

### Server_Side:
![image](https://github.com/doit-1/2/assets/136359575/27ecb638-aba9-4165-8b0d-69aa660eeb5c)


## EX-9 APPLICATION USING TCP SOCKETS - CREATING FOR CHAT CLIENT-SERVER
### Clent_Side:
```
import socket
s=socket.socket()
s.connect(('localhost',8000))
while True:
    msg=input("Client > ")
    s.send(msg.encode())
    print("Server > ",s.recv(1024).decode())
```
### Server_side:
```
import socket
s=socket.socket()
s.bind(('localhost',8000))
s.listen(5)
c,addr=s.accept()
while True:
    ClientMessage=c.recv(1024).decode()
    print("Client > ",ClientMessage)
    msg=input("Server > ")
    c.send(msg.encode())
```
### OUTPUT :
### Clent_Side:
![image](https://github.com/doit-1/2/assets/136359575/1eb79a21-a0e6-46a1-9155-2b1d2d49adf0)

### Server_side:
![image](https://github.com/doit-1/2/assets/136359575/67f748ca-9868-4b84-9818-ae36990c617c)

## EX-10 APPLICATION USING TCP SOCKETS - FILE TRANSFER PROGRAM:
### Client_Side:
```
import socket
s = socket.socket()
host = socket.gethostname()
port = 60000
s.connect((host, port))
s.send("Hello server!".encode())
with open('received_file', 'wb') as f:
    while True:
        print('receiving data...')
        data = s.recv(1024)
        print('data=%s', (data))
        if not data:
            break
        f.write(data)
f.close()
print('Successfully get the file')
s.close()
print('connection closed')
```
### Server_Side:
```
import socket
port = 60000
s = socket.socket()
host = socket.gethostname()
s.bind((host, port))
s.listen(5)
while True:
    conn, addr = s.accept()
    data = conn.recv(1024)
    print('Server received', repr(data))
    filename='mytext.txt'
    f = open(filename,'rb')
    l = f.read(1024)
    while (l):
        conn.send(l)
        print('Sent ',repr(l))
        l = f.read(1024)
    f.close()
    print('Done sending')
    conn.send('Thank you for connecting'.encode())
    conn.close()
```
### OUTPUT :
### Client_Side:
 ![image](https://github.com/doit-1/2/assets/136359575/b9b0e7fc-2e6d-4531-a144-c0049ac12e0f)

### Server_Side:
![image](https://github.com/doit-1/2/assets/136359575/2f392bfd-4c0a-4a24-9e7a-2ad3704c6425)



