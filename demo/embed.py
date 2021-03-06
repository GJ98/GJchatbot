import sys
sys.path.append('../')

from GJchatbot.model.embed.fasttext import FastText
from GJchatbot.proc.embed_processor import EmbedProcessor
from GJchatbot.utils.data_utils.dataset import Dataset

dataset = Dataset()
embed_model = FastText()
embed_processor = EmbedProcessor(embed_model)

train = True

if train == False:
    embed_processor.fit(dataset.load_embed())

while(True):
    sentence = input("입력 : ")
    if sentence == 'exit':
        exit(0)
    
    token = embed_processor.predict(sentence)
    print(token.size())
