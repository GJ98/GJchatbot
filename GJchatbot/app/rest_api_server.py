import json 
from flask import Flask, request, jsonify, abort
from GJchatbot.app.gjchatbot_api import GJchatbotApi
from GJchatbot.app.platform_transform.kakao_transform import KakaoTransform
from GJchatbot.app.platform_transform.naver_transform import NaverTransform

class RestApiServer():

    def __init__(self, api: GJchatbotApi):
        """Rest Api 서버 클래스

        Args:
            api (GJchatbotApi): 챗봇 api
        """

        self.app = Flask(__name__)
        self.api = api
        self.kakao = KakaoTransform()
        self.naver = NaverTransform()

        self.__build_route()

    def __build_route(self):
        """서버 라우트 빌드 함수"""

        @self.app.route('/query/<bot_type>', methods=['POST'])
        def query(bot_type):
            """client 질문의 답변 반환 뷰 함수

            Args:
                bot_type: client 메신저 플랫폼 
            """

            body = request.get_json()

            try:
                if bot_type == 'TEST':
                    # 챗봇 서버 테스트
                    question = body['Question']
                    answer =  self.api.get_answer(question)
                    imageUrl = None

                    if '사진보기' in answer:
                        answer, imageUrl = answer.split(' > 사진보기 : ')

                    inference = {
                        'Answer': answer,
                        'AnswerImageUrl': imageUrl
                    }
                    print("question: {}\n answer: {}, image: {}".format(question,
                                                                        answer,
                                                                        imageUrl))
                    return jsonify(inference)

                elif bot_type == 'KAKAO':
                    # 카카오톡 응답 처리(동기식)
                    question = body['userRequest']['utterance']
                    answer = self.api.get_answer(question)
                    imageUrl = None

                    if '사진보기' in answer:
                        answer, imageUrl = answer.split(' > 사진보기 : ')

                    inference = {
                        'Answer': answer,
                        'AnswerImageUrl': imageUrl
                    }   

                    kakao_format = self.kakao.json_format(inference)
                    return kakao_format
                
                elif bot_type == "NAVER":
                    # 네이버톡톡 응답 처리(비동기식)
                    user_key = body['user']
                    event = body['event']

                    if event == "open":
                        print("채팅방에 유저가 들어왔습니다")
                        return json.dumps({}), 200
                
                    elif event == "leave":
                        print("채팅방에서 유져가 나갔습니다")
                        return json.dumps({}), 200

                    elif event == "send":
                        user_text = body['textContent']['text']
                        answer = self.api.get_answer(user_text)
                        imageUrl = None

                        inference = {
                            'Answer': answer,
                            'AnswerImageUrl': imageUrl
                        }

                        naver_format = self.naver.json_format(user_key, inference)
                        self.naver.send_message(naver_format)
                        return json.dumps({}), 200


            except Exception:
                # 오류 발생 시 500 오류
                abort(500)
        