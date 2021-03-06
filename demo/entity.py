import sys
sys.path.append('../')

from GJchatbot.utils.data_utils.dataset import Dataset
from GJchatbot.model.embed.fasttext import FastText
from GJchatbot.proc.embed_processor import EmbedProcessor
from GJchatbot.proc.entity_processor import EntityProcessor
from GJchatbot.model.entity.lstm import LSTM


dataset = Dataset()
entity_dict = dataset.load_entity_dict()
embed_model = FastText()
entity_model = LSTM(entity_dict)
embed_processor = EmbedProcessor(embed_model)
entity_processor = EntityProcessor(entity_model)

tr_dl, eval_dl, label_dict = dataset.load_entity(embed_processor)
entity_processor.fit(tr_dl, eval_dl, label_dict)

while(True):
    question = input("질문 : ")
    if question == 'exit':
        exit(0)
    token = embed_processor.predict(question)
    print(entity_processor.predict(token))
