from gensim.models import FastText


class FastText(FastText):

	def __init__(self):
		"""Gensim FastText 모델 클래스"""

		vector_size = 128
		window_size = 2
		workers = 8
		min_count = 2
		iter = 2000

		super().__init__(size=vector_size,
						 window=window_size,
						 workers=workers,
						 min_count=min_count,
						 iter=iter)
