import numpy as np

import matplotlib.pyplot as plt
import os

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam




stem = "C:\\countNuclei\\RG_IncuCyte_HT1080_Images\\Phase Images\\"
stem_label = "C:\\countNuclei\\RG_IncuCyte_HT1080_Images\\Red Masks\\"

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(stem) if isfile(join(stem, f))]

#%%
imgNames = os.listdir(stem)
labelImgNames = os.listdir(stem_label)

numImages = 100
numRows = 256
numCols = 256
#X_train = np.zeros((numImages,944,1280,1))
X_train = np.zeros((numImages,numRows,numCols,1))
Y_train = np.zeros(X_train.shape)

for i in range(numImages):
    
    print(imgNames[i])
    if imgNames[i][0] == '.':
        imgName = imgNames[i][2:]
    else:
        imgName = imgNames[i]
        
    fname = stem + imgName
    img = plt.imread(fname)
    img = img[:,:,0]
    X_train[i,:,:,0] = img[0:numRows,0:numCols]
    
    labelName = imgName.replace('Phase','Red-Mask')
    fname = stem_label + labelName
    img = plt.imread(fname) 
    img = img[:,:,0]
    Y_train[i,:,:,0] = img[0:numRows,0:numCols]
    
    X_train = X_train/255.0
    Y_train = Y_train/255.0
    
#%%
#img_rows = 944
#img_cols = 1280

#img_rows = numRows
#img_cols = numCols

inputs = Input((numRows, numCols, 1))
conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(inputs)
conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv1)
pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(pool1)
conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv2)
pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(pool2)
conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv3)
pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(pool3)
conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(conv4)
pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(pool4)
conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

up6 = layers.concatenate([layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

up7 = layers.concatenate([layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

up8 = layers.concatenate([layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

up9 = layers.concatenate([layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

#%%

model.fit(X_train,Y_train, epochs = 5, batch_size = 4)

#%%
X_test = np.zeros((1,numRows,numCols,1))
#imgName = imgNames[numImages+1]
imgName = imgNames[5]
testImg = plt.imread(stem + imgName)
testImg = testImg[:,:,0]
X_test[0,:,:,0] = testImg[0:numRows,0:numCols]
X_test = X_test/255.0

Y_test = np.zeros((1,numRows,numCols,1))
labelName = imgName.replace('Phase','Red-Mask')
labelImg = plt.imread(stem_label + labelName)
labelImg = labelImg[0:numRows,0:numCols,0]
Y_test[0,:,:,0] = labelImg

#%%
check = model.predict(X_test)
predictions = check > .5

#%%
plt.figure()
plt.imshow(check[0,:,:,0])
plt.figure()
plt.imshow(predictions[0,:,:,0])
plt.figure()
plt.imshow(X_test[0,:,:,0])
plt.figure()
plt.imshow(labelImg)
    
    
    
    
    
    