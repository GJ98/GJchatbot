import os
from typing import List

import numpy as np
import torch
from torch import Tensor
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from konlpy.tag import Okt

from GJchatbot.proc.base_processor import BaseProcessor


class EmbedProcessor(BaseProcessor):

	def __init__(self, model: BaseWordEmbeddingsModel):
		"""Gensim 임베딩 모델 Training, Inference 함수를 가진 클래스

		Args:
			model (BaseWordEmbeddingsModel): Gensim 임베딩 모델	
		"""

		super().__init__(model)

		self.OOV = 1

	def fit(self, dataset: List[List[str]]):
		"""Gensim 임베딩 모델 Training 함수

		Args:
			dataset (List[List[str]]): 학습 데이터
		"""

		self.model.build_vocab(dataset)
		self.model.train(sentences=dataset,
						 total_examples=self.model.corpus_count,
						 epochs=self.model.epochs + 1)

		self.__save_model()

	def predict(self, sentence: str) -> List[List[np.ndarray]]:
		"""문자열 문장으로 임베딩 문장 반환 함수

		Args:
			sequence (str): 문자열 문장

		Returns:
			List[List[np.ndarray]]: 임베딩 문장
		 	(ex. [[token1], [token2], ...])
		"""

		self.__load_model()
		words = self.p.pos(sentence)
		return self.__forward(words)
		
	def __load_model(self):
		"""학습 모델 호출"""

		if not self.model_loaded:
			self.model_loaded = True
			self.model = self.model.__class__.load(self.model_file + '.gensim')

	def __save_model(self):
		"""학습 모델 저장"""

		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		self.model.save(self.model_file + '.gensim')

	def __forward(self, words: List[str]) -> List[List[np.ndarray]]:
		"""Gensim 임베딩 모델 Inference 함수

		Args:
			words (List[str]): 토큰화 문장

		Returns:	
			List[List[np.ndarray]]: 임베딩 문장
			(ex. [[token1], [token2], ...])
		"""
		
		sentence_vector = []

		for word in words:
			try:
				word_vector = np.array(self.model.wv[word])
			except Exception:
				word_vector = np.ones(self.model.vector_size) * self.OOV
			sentence_vector.append(word_vector)

		return sentence_vector
