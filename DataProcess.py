import os
import csv
import glob
import numpy as np
import shutil
from PIL import Image

ORIGINAL_DATA_FILE_BASE = './Original'
RIFINE_DATA_FILE_BASE = './RefineData'
X_DATA_FILE_BASE = ['./Data/X_Train', './Data/X_Valid', './Data/X_Test']
Y_DATA_FILE_BASE = ['./Data/Y_Train', './Data/Y_Valid', './Data/Y_Test']
dim = 128
""" 
1. file dictionary 를 만들고, 해당 딕셔너리로부터 csv 파일을 생성한다.
2. 생성한 딕셔너리를 기반으로 Image 이름을 1-file_num 만큼으로 전부 넘버링해서 저장한다.
3. 해당 이미지 파일과 csv의 age,gender,race 값을 가져와 npy 파일로 만들어 Input, Output을 만든다. 
"""


def parseFileName() :

	file_list = os.listdir(ORIGINAL_DATA_FILE_BASE)
	file_list_len = len(file_list)

	file_info_list = [[None,None,None,None,None] for i in range(file_list_len)]


	for idx,file in enumerate(file_list) : 
		file_name, file_ext = os.path.splitext(file)

		file_info = file_name.split('_')

		#print(file_info)
		#file_info => age, gender, race
		
		age = file_info[0]
		gender = file_info[1]
		race = file_info[2]
		date = file_info[3]

		purpose = np.random.randint(10)
		if purpose < 7 : ## 
			purpose = 0 ## 훈련용 70%
		elif 7<purpose<9 : 
			purpose = 1 ## 검증용 20%
		else :
			purpose = 2 ## 테스트용 10%

		file_info_list[idx][0] = age
		file_info_list[idx][1] = gender
		file_info_list[idx][2] = race
		file_info_list[idx][3] = date
		file_info_list[idx][4] = purpose

		if len(age) > 10 or len(gender) > 10 or len(race) > 10 :
			print("error")
			print('file : ',file)
		
	return file_info_list

def makeCSV(file_info_list) :
	f = open('./CSV/face.csv','w')
	writer = csv.writer(f)
	writer.writerow(['idx','age','gender','race','date','purpose'])
	for idx,file_info in enumerate(file_info_list) :
		csv_age = int(file_info[0])
		csv_gender = int(file_info[1])
		csv_race = int(file_info[2])
		csv_date = file_info[3]
		csv_purpose = int(file_info[4])

		writer.writerow([idx+1,csv_age,csv_gender,csv_race,csv_date,csv_purpose])
	f.close()

def refineImage(file_info_list) :

	for idx,file in enumerate(file_info_list) :
		file_name = str(file[0]) + '_' + str(file[1]) + '_' +str(file[2]) + '_' + file[3] + '.jpg'
		shutil.copy(os.path.join(ORIGINAL_DATA_FILE_BASE,file_name), os.path.join(RIFINE_DATA_FILE_BASE,'{0}.jpg'.format(idx+1)))

def makeNPY(file_info_list) :
	for idx, file in enumerate(file_info_list):
		img = Image.open(os.path.join(RIFINE_DATA_FILE_BASE,str(idx+1)+'.jpg')).resize((dim, dim))
		img = np.asarray(img)/255.0

		Y = np.array(file[0:3])
		#print(Y)
		np.save(os.path.join(X_DATA_FILE_BASE[file[4]],str(idx+1)+'.npy'),img)
		np.save(os.path.join(Y_DATA_FILE_BASE[file[4]],str(idx+1)+'.npy'),Y) 

file_info_list = parseFileName()
makeCSV(file_info_list)
#refineImage(file_info_list)
makeNPY(file_info_list)