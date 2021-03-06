import socket
import json

host = "172.30.1.29"
port = 5050

while True:
    question = input("Question : ")
    if question == 'exit':
        exit(0)
    print("="*40)

    mySocket = socket.socket()
    mySocket.connect((host, port))

    recv_json_data = {
        'Question': question,
        'BotType': "MyService"
    }

    message = json.dumps(recv_json_data)
    mySocket.send(message.encode())

    data = mySocket.recv(2048).decode()
    send_json_data = json.loads(data)
    print("Answer : ")
    print(send_json_data['Answer'],'\n')

mySocket.close()