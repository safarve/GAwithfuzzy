import cv2
import numpy as np
import pathlib
import os
import glob
import math
from GA import FuzzyPatternClassifierGA
from sklearn.model_selection import train_test_split

#Preprocesing steps, Output --> wirtes procesed images to the folder Preprocessed
def preprocess(filename,i):
	img=cv2.imread(filename)
	ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	dst = cv2.fastNlMeansDenoisingColored(thresh1,None, h=5)
	blur = cv2.GaussianBlur(dst,(5,5),0)
	idx = np.argwhere(np.all(blur[..., :] >250, axis=0))
	a2 = np.delete(blur, idx, axis=1)
	idx = np.argwhere(np.all(blur[:,...] >250, axis=1))
	a3 = np.delete(a2,idx,axis=0)
	w=32
	h=42
	resized = cv2.resize(a3, (w,h), interpolation = cv2.INTER_AREA) 
	path =os.path.join(pathlib.Path(__file__).parent.absolute(),'Preprocessed')
	cv2.imwrite(os.path.join(path,str(i)+'.png'),resized)

#Split training and testing data, split == 67-33
def traintestsplit(X,y):
	i=0;
	X_train,X_test,y_train,y_test=train_test_split(features[i*55:i*55+55], label[i*55:i*55+55], test_size=0.33, random_state=42)
	rge = [i for i in range(10+26+26) if i!=0]
	for i in rge:
		X_traintemp, X_testtemp, y_traintemp, y_testtemp = train_test_split(features[i*55:i*55+55], label[i*55:i*55+55], test_size=0.33, random_state=42)
		X_train=np.concatenate((X_train,X_traintemp))
		y_train=np.concatenate((y_train,y_traintemp))
		X_test=np.concatenate((X_test,X_testtemp))
		y_test=np.concatenate((y_test,y_testtemp))
	return(X_train,X_test,y_train,y_test)

# Uncomment only for frist run, or want to see preprocessing, computationally intensive
# path = glob.glob(os.path.join(pathlib.Path(__file__).parent.absolute(),'EnglishHnd\English\Hnd\Img\*'))
# i=0;
# for folder in path:
#     for img in glob.glob(folder+'/*.png'):
#         preprocess(img,i)
#         if(i%10==0):
#         	print("number of images processed",i)
#         i=i+1
        

#feature extraction
features=np.empty((3410,168),float)
path = glob.glob('Preprocessed\*.png')
f_index=0;

for image in path:
	img=cv2.imread(image)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_box=img_gray.reshape(6,4,7,8)
	sigma = np.full(24,0.0)
	
	for i in range(6):
		for j in range(4):
			sum=0
			angle=0;
			for l in range(7):
				for k in range(8):
					if(img_box[i,j,l,k]<50):
						sum=sum+math.sqrt(l^2+k^2)
			sigma[i*4+j]=(1/56.0)*sum
			
	mean = np.empty(24,float)
	for i in range(6):
		for j in range(4):
			mean[i*4+j] = (1/56.0)*img_box[i,j].sum()
	
	diag=np.empty(24,float)
	for i in range(6):
		for j in range(4):
			diag[i*4+j] = (1/7.0)*img_box[i,j].trace()
	sd=np.empty((24),float)
	for i in range(6):
		for j in range(4):
			sd[i*4+j] = np.std(img_box[i,j])

	xcent=np.empty(24,int)
	ycent=np.empty(24,int)
	for i in range(6):
		for j in range(4):
			contours, hierarchy = cv2.findContours(img_box[i,j], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
			kpCnt = len(contours[0])
			x = 0
			y = 0

			for kp in contours[0]:
	  			x = x+kp[0][0]
	  			y = y+kp[0][1]
			xcent[4*i+j]=np.ceil(x/kpCnt)
			ycent[4*i+j]=np.ceil(y/kpCnt)
	grad=np.empty(24,float)
	for k in range(6):
		for j in range(4):
			sum=0
			sobelx = cv2.Sobel(img_box[k,j],cv2.CV_64F,1,0,ksize=5).reshape(56)
			sobely = cv2.Sobel(img_box[k,j],cv2.CV_64F,0,1,ksize=5).reshape(56)
			for i in range(56):
				sum=sum+math.sqrt(sobelx[i]*sobelx[i]+sobely[i]*sobely[i])
			grad[4*k+j]=sum
	features[f_index]=np.concatenate((sigma,mean,diag,sd,xcent,ycent,grad))
	f_index=f_index+1

label=np.array([])
for i in range(10+26+26):
	label=np.concatenate((label,np.full(55,i)))

X_train, X_test, y_train, y_test = traintestsplit(features,label)

c=FuzzyPatternClassifierGA()
model = c.fit(X_train,y_train)
testlabels = model.predict(X_test)

#writing results
f = open('resultsGAplusFuzzy.dat',"w")
j=0
err=0
for i in testlabels:
	f.write(str(i)+',')
	f.write(str(y_test[j])+'\n')
	
	if(i!=y_test[j]):
		err= err+1
	j=j+1
print('Accuracy=',100*(1-(err/(len(testlabels)))))
f.close()


