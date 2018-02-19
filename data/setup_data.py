import pandas as  pd
import cv2
import os
from PIL import Image
from matplotlib import cm
import time
import numpy as np

nclasses = [(0,'Angry'), (1,'Disgust'), (2,'Fear'), (3,'Happy'), (4,'Sad'), (5,'Surprise'), (6,'Neutral')]

def create_class_directories(dirtype,nclasses):
    if not os.path.exists(dirtype):
        os.makedirs(dirtype)
        #(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
        for idx,name in nclasses:
            os.makedirs(f'./{dirtype}/{name}-{idx}')
            
def save_image(img,dirtype,label):
    img = cv2.resize(img.astype('float'),(200,200))
    img3c = np.zeros((200,200,3))
    img3c[:,:,0] = img
    img3c[:,:,1] = img
    img3c[:,:,2] = img
#     print(img3c.shape)
    img = Image.fromarray(img3c.astype('uint8'), 'RGB')
    img.save(f'./{dirtype}/{nclasses[label][1]}-{label}/out{time.time()}.bmp')    
    

for i in ['train','val','test']:
    create_class_directories(i,nclasses)

print('Extracting data...\n')

df = pd.read_csv('train.csv')
df_train = df[df['Usage'] == 'Training']
df_val = df[df['Usage'] == 'PublicTest']
df_test = df[df['Usage'] == 'PrivateTest']

print('writing train data to file...(X_train.csv/y_train.csv)...and images to directories')
with open('X_train.csv','w') as file:
    for index, ex in df_train.iterrows():
        e = ex['pixels']
        lb = ex['emotion']
        img = np.array(e.split()).reshape(48,48)
#         print(img.shape)
        save_image(img,'train',lb)  
#         exit()
        print(','.join(e.split()),file=file)
with open('y_train.csv','w') as file:
    for e in df_train['emotion']:
        print(e,file=file)

print('writing validation data to file...(X_val.csv/y_val.csv)')
with open('X_val.csv','w') as file:
    for index, ex in df_val.iterrows():
        e = ex['pixels']
        lb = ex['emotion']
        img = np.array(e.split()).reshape(48,48)
#         print(img.shape)
        save_image(img,'val',lb)  
#         exit()
        print(','.join(e.split()),file=file)
with open('y_val.csv','w') as file:
    for e in df_val['emotion']:
        print(e,file=file)

print('writing test data to file...(X_test.csv/y_test.csv)')
with open('X_test.csv','w') as file:
    for index, ex in df_test.iterrows():
        e = ex['pixels']
        lb = ex['emotion']
        img = np.array(e.split()).reshape(48,48)
#         print(img.shape)
        save_image(img,'test',lb)  
#         exit()
        print(','.join(e.split()),file=file)
with open('y_test.csv','w') as file:
    for e in df_test['emotion']:
        print(e,file=file)

print('\n\ndone!')