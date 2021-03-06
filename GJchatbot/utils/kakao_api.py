import requests
import json
from random import randrange


class KakaoApi:

	def __init__(self):
		"""kakao rest api 클래스"""

		self.url  = 'https://dapi.kakao.com/v2/local/search/'
		self.headers = {
    		"Authorization": "KakaoAK df8d3209a7ab037393012ec2cf508a91"
		}
		self.category = None

	def request(self, intent: str, location: str, place: str) -> str:
		"""local rest api 함수

		Args:
			intent (str): 의도
			location (str): 위치
			place (str): 장소

		Returns:
			str: 답변
		"""

		self.category = ("FD6", '맛집') if intent == 'restaurant' else ("AT4", '관광 명소')

		location_url = self.url + 'address.json?query={}'.format(location)
		req = requests.get(location_url, headers=self.headers).json()['documents'][0]

		x, y = req['x'], req['y']

		place_url = (self.url + 
					'category.json?category_group_code={0}&x={1}&y={2}&radius=20000'
					.format(self.category[0], x, y))
		request = requests.get(place_url, headers=self.headers).json()['documents']
		idx = randrange(len(request))
		req = request[idx]

		ans = ("{0}의 {1}에 대한 정보를 전해드릴게요! 😀😀\n"
			  "{0} 20km 반경 내 {2}과 관련된 {3}에 가보시는 건 어떤가요?\n"
			  "주소는 {4}입니다. 장소 상세 페이지 URL : {5}".format(location,
				  													self.category[1],
																	req['category_name'], 
																	req['place_name'], 
																	req['road_address_name'], 
																	req['place_url']))
	
		return ans
