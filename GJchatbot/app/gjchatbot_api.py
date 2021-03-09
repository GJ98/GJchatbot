from typing import List

from kocrawl.dust import DustCrawler
from kocrawl.weather import WeatherCrawler
from kocrawl.map import MapCrawler

from GJchatbot.utils.data_utils.preprocessor import Preprocessor
from GJchatbot.utils.data_utils.dataset import Dataset
from GJchatbot.proc.embed_processor import EmbedProcessor
from GJchatbot.proc.intent_processor import IntentProcessor
from GJchatbot.proc.entity_processor import EntityProcessor
from GJchatbot.utils.kakao_api import KakaoApi

class GJchatbotApi:

    def __init__(self, 
                 dataset: Dataset, 
                 embed_processor: EmbedProcessor, 
                 intent_processor: IntentProcessor, 
                 entity_processor: EntityProcessor, 
                 train: bool):
        """챗봇 API 클래스

        Args:
            dataset (Dataset): 데이터
            embed_processor (EmbedProcessor): 임베딩 프로세서
            intent_processor (IntentProcessor): 의도 분류 프로세서
            entity_processor (EntityProcessor): 개채명 분류 프로세서
            train (bool): 학습 여부
        """

        self.embed_processor = embed_processor
        self.intent_processor = intent_processor
        self.entity_processor = entity_processor
        self.p = Preprocessor()
        self.kakao_api = KakaoApi()
        self.location, self.date, self.place = [], [], []
        self.intent = None

        if train is True:
            #self.embed_processor.fit(dataset.load_embed())
            dl1, dl2, label_dict = dataset.load_intent(self.embed_processor)
            self.intent_processor.fit(dl1, dl2, label_dict)
            dl1, dl2, label_dict = dataset.load_entity(self.embed_processor)
            self.entity_processor.fit(dl1, dl2, label_dict)

    def get_answer(self, question: str):
        """질문 의도 및 개채명 추론을 통한 답변 반환 함수

        Args:
            question (str): 문자열 문장

        Returns:
            str: 추론 답변
        """
        
        words = self.p.pos(question)
        token = self.embed_processor.predict(question)
        entity = self.entity_processor.predict(token)
        if self.intent is None:
            self.intent = self.intent_processor.predict(token)

        print('intent : ', self.intent)
        print('entity : ', entity)

        for idx, e in enumerate(entity):
            if "LOCATION" in e: self.location.append(words[idx])
            elif "DATE" in e: self.date.append(words[idx])
            elif "PLACE" in e: self.place.append(words[idx])
        if len(self.date) == 0:
            self.date.append('오늘')
        if len(self.place) == 0 and self.intent == 'restaurant':
            self.place.append('맛집')
        if len(self.place) == 0 and self.intent == 'travel':
            self.place.append('관광지')

        if len(self.location) == 0:
            return "어느 지역을 알려드릴까요"

        print('location: {}, date: {}, place: {}'.format(self.location, self.date, self.place))

        if self.intent == 'weather':
            crawler = WeatherCrawler()
            ans = crawler.request(location=" ".join(self.location),
                                  date=" ".join(self.date))
        elif self.intent == 'dust':
            crawler = DustCrawler()
            ans = crawler.request(location=" ".join(self.location),
                                  date=" ".join(self.date))
        else:
            ans = self.kakao_api.request(intent=self.intent, 
                                         location=" ".join(self.location),
                                         place=" ".join(self.place))
        
        self.location, self.date, self.place = [], [], []
        self.intent = None

        return ans

    def get_intent(self, question: str) -> str:
        """질문 의도 추론 함수

        Args:
            question (str): 질문 문자열 문장
        
        Returns:
            str: 추론 의도
        """

        token = self.embed_processor.predict(question)
        intent = self.intent_processor.predict(token)

        return intent

    def get_entity(self, question: str) -> List[str]:
        """질문 개채명 추론 함수

        Args:
            question (str): 질문 문자열 문장

        Returns:
            List[str]: 추론 개채명 리스트
        """

        token = self.embed_processor.predict(question)
        entity = self.entity_processor.predict(token)

        return entity

