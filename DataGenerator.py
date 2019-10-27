from tensorflow import keras
import numpy as np
import os

class DataGenerator(keras.utils.Sequence) :
	## 배치 데이터 생성용
	def __init__(self, input_x, output_y, batch_size = 32, dim=(224,224), n_channels=3, n_classes = None, shuffle=True):
		self.input_x = input_x
		self.output_y = output_y
		self.batch_size = batch_size
		self.dim = dim
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self) :
		return int(np.floor(len(self.input_x)/self.batch_size))

	def __getitem__(self, index) :
		indices = self.indices[index*self.batch_size:(index+1)*self.batch_size] ## 전체 데이터에서 배치로 사용할 범위의 인덱스 지정
		input_x = [self.input_x[k] for k in indices] ## 입력 데이터들의 리스트에서 batch_size 만큼의 이미지 ID를 가져옴
		output_y = [self.output_y[k] for k in indices]
		X,y = self.__data_generation(input_x,output_y)

		return X,y

	def on_epoch_end(self) :
		self.indices = np.arange(len(self.input_x))

		if self.shuffle == True :
			np.random.shuffle(self.indices)

	def __data_generation(self,input_x, output_y) :
		X = np.empty((self.batch_size, *self.dim, self.n_channels)) ## self.dim = 이미지 해상도, channels = 이미지 채널 수 (RGB = 3)
		y = np.empty((self.batch_size, 1)) ## age 는 한자리이므로 1로 표기

		for idx,(input_ID,output_ID) in enumerate(zip(input_x,output_y)) : ## i 는 파일 저장 경로가 포함된 파일 명
			X[idx] = np.load(input_ID)
			temp_y = np.load(output_ID)
			y[idx] = temp_y[0] ## 0번쨰가 나이 col
		return X,y
