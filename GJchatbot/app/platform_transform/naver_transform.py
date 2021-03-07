import requests, json
from typing import Dict


class NaverTransform:

    def __init__(self):
        """네이버 톡톡 응답 JOSN 포맷 전환 클래스"""

        self.authorization_key = '3z1WoQAbQ8Cv1ZMl8DO9'

    def text_format(self, text: str) -> Dict[str, any]:
        """네이버 톡톡 텍스트 JSON 포맷 전환 함수

        Args:
            text (str): 텍스트

        Returns:
            Dict[str, any]: 네이터 톡톡 문장 JSON 포맷인 텍스트
        """

        return {
            "textContent": {
                "text": text
            }
        }

    def image_format(self, imageUrl: str) -> Dict[str, any]:
        """네이버 톡톡 이미지 JSON 포맷 전환 함수

        Args:
            imageUrl (str): 이미지 URL

        Returns:
            Dict[str, any]: 네이버 톡톡 이미지 JSON 포맷인 이미지
        """

        return {
            "imageContent": {
                "imageUrl": imageUrl
            }
        }

    def json_format(self, user_key: str, bot_resp: Dict[str, str]):
        """네이버 톡톡 응답 JSON 포맷 전환 함수

        Args:
            user_key (str): 유저 키
            bot_resp (Dict[str, str]): 챗봇 추론값

        Returns:
            Dict[str,str]: 네이버 톡톡 응답 JSON 포맷인 추론값
        """

        json_format = {
            "event": "send",
            "user": user_key,
        }

        if bot_resp['AnswerImageUrl'] is not None:
            json_format.update(
                self.image_format(bot_resp['AnswerImageUrl'])
            )
        
        if bot_resp['Answer'] is not None:
            json_format.update(
                self.text_format(bot_resp['Answer'])
            )
        
        return json_format

    def send_message(self, naver_format: Dict[str, any]):
        """rest api 방식으로 답변 전달 함수

        Args:
            naver_format (Dict[str, any]): 네이버 톡톡 응답 JSON 포맷인 답변
        """

        headers = {
            'Content-Type': 'application/json;charset=UTF-8',
            'Authorization': self.authorization_key,
        }

        message = json.dumps(naver_format)
        return requests.post('https://gw.talk.naver.com/chatbot/v1/event',
                             headers=headers,
                             data=message)
        



