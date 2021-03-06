import sys
sys.path.append('../')

from GJchatbot.utils.data_utils.dataset import Dataset
from GJchatbot.proc.embed_processor import EmbedProcessor
from GJchatbot.model.embed.fasttext import FastText

dataset = Dataset()
embed_model = FastText()
embed_processor = EmbedProcessor(embed_model)


dataset.load_intent(embed_processor)
dataset.load_entity(embed_processor)
