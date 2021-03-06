import sys
sys.path.append('../')

from GJchatbot.utils.data_utils.dataset import Dataset
from GJchatbot.proc.embed_processor import EmbedProcessor
from GJchatbot.model.embed.fasttext import FastText
from GJchatbot.proc.intent_processor import IntentProcessor
from GJchatbot.model.intent.cnn import CNN

dataset = Dataset()
intent_dict = dataset.load_intent_dict()
embed_model = FastText()
intent_model = CNN(intent_dict)
embed_processor = EmbedProcessor(embed_model)
intent_processor = IntentProcessor(intent_model)

train_dataloader, test_dataloader, label_list = dataset.load_intent(embed_processor)
intent_processor.fit(train_dataloader, test_dataloader, label_list)

while(True):
    question = input("질문 : ")
    if question == 'exit':
        exit(0)
    token = embed_processor.predict(question)
    print(intent_processor.predict(token))

