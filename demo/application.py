import sys
sys.path.append('../')

from GJchatbot.app.gjchatbot_api import GJchatbotApi
from GJchatbot.utils.data_utils.dataset import Dataset
from GJchatbot.model.embed.fasttext import FastText
from GJchatbot.model.intent.cnn import CNN
from GJchatbot.model.entity.lstm import LSTM
from GJchatbot.proc.embed_processor import EmbedProcessor
from GJchatbot.proc.intent_processor import IntentProcessor
from GJchatbot.proc.entity_processor import EntityProcessor
from GJchatbot.app.rest_api_server import RestApiServer

dataset = Dataset()
intent_dict = dataset.load_intent_dict()
entity_dict = dataset.load_entity_dict()

embed_processor = EmbedProcessor(model=FastText())
intent_processor = IntentProcessor(model=CNN(intent_dict))
entity_processor = EntityProcessor(model=LSTM(entity_dict))

api = GJchatbotApi(dataset=Dataset(),
                   embed_processor=embed_processor,
                   intent_processor=intent_processor,
                   entity_processor=entity_processor,
                   train=False)

server = RestApiServer(api)

server.app.run(host='0.0.0.0', port=5000)
