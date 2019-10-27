from MyResnet import myModel
import os
import numpy as np
import cv2
import tensorflow as tf

def resizeImage(image, size) :
	image = tf.image.resize(image, (size,size))
	image = image / 255

	return image

my_model = myModel()

my_model.load_weights('./models/model.h5')
my_model.compile(optimizer='adam', loss='mse')

file_list = os.listdir('./Data/X_Test')
answer_list = os.listdir('./Data/Y_Test')

x_base = './Data/X_Test'
y_base = './Data/Y_Test'
origin_image_base = './RefineData'

for file,answer in zip(file_list,answer_list): 
	file_name,file_ext = os.path.splitext(file)
	origin_img = cv2.imread(os.path.join(origin_image_base,file_name+'.jpg'))
	x = np.load(os.path.join(x_base,file))
	y = np.load(os.path.join(y_base,answer))
	x_resize = x.reshape((1,128,128,3))

	#print(x_resize.shape)
	y_predict = my_model.predict(x_resize)
	y_predict = y_predict.squeeze()
	print('y_predict  : ', y_predict, ' shape : ', y_predict.shape )
	cv2.putText(origin_img, "true : {0} pred : {1}".format(y[0],int(y_predict)),(0,20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,(255,0,0),1)

	cv2.imshow('x',origin_img)
	k = cv2.waitKey()

	if k == ord('q') :
		cv2.destroyAllWindows()
		break

cv2.destroyAllWindows()

