from flask import Flask, request
app = Flask(__name__)

@app.route('/api/sayHello', methods=['POST'])
def sayHello():
	body = request.get_json()
	print(body)

	responseBody = {
		"version": "2.0",
		"template": {
			"outputs": [
				{
					"simpleText": {
						"text": "안녕 hello I'm Ryan"
					}
				}
			]
		}
	}
	return responseBody

@app.route('/api/showHello', methods=['POST'])
def showHello():
	body = request.get_json()
	print(body)

	responseBody = {
		"version": "2.0",
		"template": {
			"outputs": [
				{
					"simpleImage": {
						"imageUrl": "https://t1.daumcdn.net/friends/prod/category/M001_friends_ryan2.jpg",
						"altText": "hello I'm Ryan"
					}
				}
			]
		}
	}
	return responseBody

if __name__=='__main__':
	app.run(host='127.0.0.1', port=5000)
