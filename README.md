# Resnet 50 기반 나이 예측
----------------------------------
* Deep Learning 구조를 실제 구현하는 공부를 하며 Tensorflow 2.0 을 사용  
Resnet 50을 구현하고 신경망을 UTK_Face 데이터셋 1만개 데이터를 이용하여 훈련  
나이를 예측하기 위한 얼굴 상의 피쳐를 뽑아내도록 훈련시킴.  
후에 얼굴 인식 프로젝트를 별도 진행해보기 위한 전초 작업. 

# Training
-----------
* [UTK Face](https://www.kaggle.com/jangedoo/utkface-new) 를 이용하여 훈련  
1만개의 얼굴 데이터 중 70%를 훈련용으로 사용했으며 20%는 검증용 나머지 10%는 테스트용으로 사용  
* 한계점 - 고령자의 경우 정확한 나이 측정은 하지 못함. 대략적으로 고령이다 정도만 파악하는 수준  

# Result
---------

![result2.png](https://github.com/elensar92/DNN_AGE_PREDICTOR/blob/master/Result/result2.png?raw=true)
![result3.png](https://github.com/elensar92/DNN_AGE_PREDICTOR/blob/master/Result/result3.png?raw=true)
![result6.png](https://github.com/elensar92/DNN_AGE_PREDICTOR/blob/master/Result/result6.png?raw=true)

# Source
--------
* MyResnet.py  
Resnet 50을 실제 코드로 구현하여 모델을 반환하는 Python 파일
* Training.py  
학습 데이터로 신경망 학습 
* DataGenerator.py  
배치 생성
* DataProcess.py  
UTK_Face 파일 1만개를 Train, Valid, Test 용으로 분리.
* Test.py  
샘플 데이터로 결과를 보기 위한 파일

# Enviroment
____________-
* Linux Ubuntu 18.04
* Mac OS X

# Dependencies
--------------
* Tensorflow 2.0
* Python 3.7
* PyPL
* OpenCV
* Numpy
* glob, shutils etc
