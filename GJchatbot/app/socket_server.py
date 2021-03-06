import json
import socket

from GJchatbot.app.gjchatbot_api import GJchatbotApi
from GJchatbot.utils.data_utils.dataset import Dataset
from GJchatbot.model.embed.fasttext import FastText
from GJchatbot.model.intent.cnn import CNN
from GJchatbot.model.entity.lstm import LSTM
from GJchatbot.proc.embed_processor import EmbedProcessor
from GJchatbot.proc.intent_processor import IntentProcessor
from GJchatbot.proc.entity_processor import EntityProcessor

class socketServer:

    def __init__(self, srv_port, listen_num, api):

        self.api = api
        self.port = srv_port
        self.listen = listen_num
        self.mySock = None

    def create_server(self):
        self.mySock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mySock.bind(("0.0.0.0", int(self.port)))
        self.mySock.listen(int(self.listen))
        print("create server(port: 5050)")

    def start_service(self):
        print("start service")
        while(True):
            conn, addr = self.mySock.accept()

            read = conn.recv(2048)
            print('='*30)
            print('Connection from: {}' .format(addr))

            if read is None or not read:
                print('클라이언트 연결 끊어짐')
                conn.close()

            recv_json_data = json.loads(read.decode())
            question = recv_json_data['Question']
            answer = self.api.get_answer(question)

            send_json_data = {
                'Question': question,
                'Answer': answer
            }
            message = json.dumps(send_json_data)
            conn.send(message.encode())
            conn.close()

