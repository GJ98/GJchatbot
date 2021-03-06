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
        intent = self.intent_processor.predict(token)

        print('intent : ', intent)
        print('entity : ', entity)

        location, date, place = [], [], []
        for idx, e in enumerate(entity):
            if "LOCATION" in e: location.append(words[idx])
            elif "DATE" in e: date.append(words[idx])
            elif "PLACE" in e: place.append(words[idx])
        if len(date) == 0:
            date.append('오늘')
        if len(place) == 0 and intent == 'restaurant':
            place.append('맛집')
        if len(place) == 0 and intent == 'travel':
            place.append('관광지')

        ans = "잘 모르겠어요."
        if intent == 'weather':
            crawler = WeatherCrawler()
            ans = crawler.request(location=" ".join(location),
                                  date=" ".join(date))
        elif intent == 'dust':
            crawler = DustCrawler()
            ans = crawler.request(location=" ".join(location),
                                  date=" ".join(date))
        else:
            ans = self.kakao_api.request(intent=intent, 
                                         location=" ".join(location),
                                         place=" ".join(place))
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

