from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from GJchatbot_config import PATH 
from GJchatbot.proc.embed_processor import EmbedProcessor
from GJchatbot.utils.data_utils.preprocessor import Preprocessor


class Dataset:

	def __init__(self):
		"""학습 및 검증 데이터와 라벨 딕셔너리 생성 함수를 가진 클래스"""

		self.p = Preprocessor()
		self.batch_size = 512
		self.data_ratio = 0.8

	def load_embed(self) -> List[List[str]]:
		"""임베딩 프로세서 학습 데이터 생성 함수

		Returns:
			List[List[str]]: 임베딩 프로세서 학습 데이터
			(ex.[[token1]. [token2], [token3], ...])
		"""

		intent_data_dir = PATH['intent_data_dir']

		dataset = pd.read_csv(intent_data_dir, delimiter=',')

		question = list(dataset['question'])

		# 토큰화 
		for idx, q in enumerate(question):
			question[idx] = q.split()

		return question

	def load_intent(self, embed_processor: EmbedProcessor):
		"""의도 분류 프로세서 학습 및 검증 데이터 생성 함수

		Args:
			embed_processor (EmbedProcessor): 임베딩 프로세서

		Returns:
			DataLoader: 의도 학습 데이터
			DataLoader: 의도 검증 데이터
			Dict[int, str]: 의도 라벨 딕셔너리
		"""

		intent_data_dir = PATH['intent_data_dir']

		dataset = pd.read_csv(intent_data_dir, delimiter=',')

		question, label = list(dataset['question']), list(dataset['label'])

		label2idx, idx2label = self.__make_label_dict([label])
		print("label_dict\n", label2idx)

		for idx, (q, l) in tqdm(enumerate(zip(question, label)), 
								desc='preprocessing', 
								total=len(question)):
			# question 임베딩 
			q = embed_processor.predict(q)
			question[idx] = self.p.pad_question(q)
			# question: [[q_token1], [q_token2], ...]

			# label 엠베딩
			label[idx] = label2idx[l]
			# label: label

		dataset = TensorDataset(torch.tensor(question), torch.tensor(label))

		# 학습 데이터, 검증 데이터 분할 
		split_point = int(len(dataset) * self.data_ratio)
		train_dataset, test_dataset = \
			random_split(dataset, [split_point, len(dataset) - split_point])

		train_dataloader = DataLoader(dataset=train_dataset,
									  batch_size=self.batch_size,
									  shuffle=True,
									  pin_memory=True)

		test_dataloader = DataLoader(dataset=test_dataset,
									  batch_size=self.batch_size,
									  shuffle=True,
									  pin_memory=True)

		return train_dataloader, test_dataloader, idx2label

	def load_entity(self, embed_processor: EmbedProcessor):
		"""개채명 분류 프로세서 학습 및 검증 데이터 생성 함수

		Args:
			embed_prcessor (EmbedProcessor): 임베딩 프로세서

		Returns:
			Dataloader: 학습 데이터
			Dataloader: 검증 데이터
			Dict[int, str]: 개채명 라벨 딕셔너리
		"""

		entity_data_dir = PATH['entity_data_dir']

		dataset = pd.read_csv(entity_data_dir, delimiter=',')

		question, label = list(dataset['question']), list(dataset['label'])

		for idx, l in enumerate(label):
			label[idx] = l.split()

		label2idx, idx2label = self.__make_label_dict(label)
		print("label_dict\n", label2idx)

		for idx, (Q, L) in tqdm(enumerate(zip(question, label)), 
								desc='preprocessing', 
								total=len(question)):
			# question 임베딩 
			Q = embed_processor.predict(Q)
			question[idx] = self.p.pad_question(Q)
			# question: [[q_token1], [q_token2], ...]

			# label 엠베딩
			for i, l in enumerate(L):
				L[i] = label2idx[l]
			label[idx] = self.p.pad_label(L, label2idx)
			# label: [q_label1, q_label2, ...]

		dataset = TensorDataset(torch.tensor(question), torch.tensor(label))

		# 학습 데이터, 검증 데이터 분할
		split_point = int(len(dataset) * self.data_ratio)
		train_dataset, test_dataset = \
			random_split(dataset, [split_point, len(dataset) - split_point])

		train_dataloader = DataLoader(dataset=train_dataset,
									  batch_size=self.batch_size,
									  shuffle=True,
									  pin_memory=True)

		test_dataloader = DataLoader(dataset=test_dataset,
									  batch_size=self.batch_size,
									  shuffle=True,
									  pin_memory=True)

		return train_dataloader, test_dataloader, idx2label

	def load_intent_dict(self) -> Dict[int, str]:
		"""의도 라벨 딕셔너리 생성 함수

		Returns
			Dict[int, str]: 의도 라벨 딕셔너리
		"""

		intent_data_dir = PATH['intent_data_dir']

		dataset = pd.read_csv(intent_data_dir, delimiter=',')

		label = list(dataset['label'])

		_, idx2label = self.__make_label_dict([label])

		return idx2label
		

	def load_entity_dict(self) -> Dict[int, str]:
		"""개채명 라벨 딕셔너리 생성 함수

		Returns
			Dict[int, str]: 개채명 라벨 딕셔너리
		"""

		entity_data_dir = PATH['entity_data_dir']

		dataset = pd.read_csv(entity_data_dir, delimiter=',')

		label = list(dataset['label'])

		for idx, l in enumerate(label):
			label[idx] = l.split()

		_, idx2label = self.__make_label_dict(label)

		return idx2label
	

	def __make_label_dict(self, labels):
		"""라벨 딕셔너리 생성 함수

		Args:
			labels (List[List[str]]): 라벨 데이터

		Returns:
			Dict[str, int]: lable->idx 딕셔너리
			Dict[int, str]: idx->label 딕셔너리
		"""

		idx = 0
		label2idx, idx2label = {}, {}
		for label in labels:
			for l in label:
				if l not in label2idx:
					label2idx[l] = idx
					idx2label[idx] = l
					idx += 1

		return label2idx, idx2label
