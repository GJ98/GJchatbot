from typing import List, Dict
from konlpy.tag import Okt

import numpy as np

class Preprocessor:

    def __init__(self):
        """데이터 전처리 함수를 가진 클래스"""

        self.tokenizer = Okt()
        self.maxlen = 8
        self.vector_size = 128

        self.PAD = 0
        self.NER_outside = 'O'

    def pos(self, sentence: str) -> List[str]:
        """문장을 토큰화 해주는 함수

        Args:
            sentence (str): 문자열 문장
        
        Returns:
            List[str]: 토큰화 문장
        """

        words = []
        for word, pos in self.tokenizer.pos(sentence):
            if pos not in ['Josa', 'Punctuation']:
                words.append(word)
        return words

    def pad_question(self, words: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """문장 길이를 고정하는 패딩 함수

        Args:
            words (List[List[np.ndarray]]): 임베딩 문장

        Returns:
            List[List[np.ndarray]]: 패딩 문장
        """

        length = len(words)

        if self.maxlen < length:
            words = words[:self.maxlen]
        else:
            pad = np.ones([self.maxlen, self.vector_size]) * self.PAD
            for idx in range(length):
                pad[idx] = words[idx]
            words = pad

        return words

    def pad_label(self, labels: List[int], label_dict: Dict[str, int]) -> List[int]:
        """라벨 문장 길이를 고정하는 패딩 함수

        Args:
            labels (List[int]): 임베딩 라벨 문장
            label_dict (Dict[str, int]): 라벨 딕셔너리

        Returns:
            List[int]: 패딩 라벨 문장
        """

        length = len(labels)

        if self.maxlen < length:
            labels = labels[:self.maxlen]
        else:
            pad = np.ones([self.maxlen]) * label_dict[self.NER_outside]
            for idx in range(length):
                pad[idx] = labels[idx]
            labels = pad

        return labels