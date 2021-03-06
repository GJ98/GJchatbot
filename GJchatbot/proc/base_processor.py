from typing import Any
from abc import abstractmethod

from GJchatbot_config import PATH
from GJchatbot.utils.data_utils.preprocessor import Preprocessor


class BaseProcessor:

	def __init__(self, model: Any):
		"""모든 프로세서의 부모 클래스

		Args
			model (Any): 학습 모델
		"""

		super().__init__()

		self.p = Preprocessor()
		self.delimeter = PATH['delimeter']
		self.model_dir = PATH['model_dir']

		self.model = model
		self.model_loaded = False

		# /saved/CLASS_NAME/
		self.model_dir = (self.model_dir + 
						  self.__class__.__name__  +
						  self.delimeter)

		# /saved/CLASS_NAME/CLASS_NAME.xxx
		self.model_file = (self.model_dir + 
						   self.__class__.__name__)

	@abstractmethod
	def fit(self, *args, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def predict(self, *args, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def __load_model(self):
		raise NotImplementedError

	@abstractmethod
	def __save_model(self):
		raise NotImplementedError

