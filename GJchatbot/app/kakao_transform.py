from typing import Dict


class KakaoTransform:

    def __init__(self):
        """카카오톡 응답 JSON 포맷 전환 함수를 가진 클래스"""

        self.version = '2.0'

    def text_format(self, text: str) -> Dict[str, any]:
        """단순 텍스트 JSON 포맷 전환 함수

        Args:
            text (str): 텍스트

        Returns:
            Dict[str, any]: 카톡 문장 JSON 포맷인 텍스트
        """

        return {
            "simpleText": {"text": text}
        }

    def image_format(self, imageUrl: str, altText: str) -> Dict[str, any]:
        """ 단순 이미지 JSON 포맷 전환 함수

        Args:
            imageUrl (str): 이미지 URL
            altText (str): 대체 텍스트

        Returns:
            Dict[str, any]: 카톡 이미지 JSON 포맷인 이미지
        
        """

        return {
            "simpleImage": {
                "imageUrl": imageUrl, 
                "altText": altText
            }
        }

    def json_format(self, bot_resp: Dict[str, str]) -> Dict[str, any]:
        """카카오톡 응답 JSON 포맷 전환 함수

        Args:
            bot_resp (Dict[str, str]): 챗봇 추론값

        Returns:
            Dict[str, any]: 카톡 응답 JSON 포맷인 추론값
        """

        json_format = {
            "version": self.version,
            "template": {
                "outputs": []
            }
        }

        if bot_resp['AnswerImageUrl'] is not None:
            json_format['template']['outputs'].append(
                self.image_format(bot_resp['AnswerImageUrl'], '')
            )

        if bot_resp['Answer'] is not None:
            json_format['template']['outputs'].append(
                self.text_format(bot_resp['Answer'])
            )
        
        return json_format
        