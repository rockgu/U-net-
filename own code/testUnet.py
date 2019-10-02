
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam

#%% U-net architecture
img_rows = 512
img_cols = 512

inputs = Input((img_rows, img_cols, 1))
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

model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])


#%% loading the training data
import matplotlib.pyplot as plt

stem = "/Users/rockgu/Documents/U-net/unet/data/membrane/train/image/"
stem_label = "/Users/rockgu/Documents/U-net/unet/data/membrane/train/label/"

numImages = 1
X_train = np.zeros((numImages,512,512,1))
Y_train = np.zeros(X_train.shape)

for i in range(numImages):
    
    fname = stem + str(i) +  ".png"
    img = plt.imread(fname)    
    X_train[i,:,:,0] = img
    
    fname = stem_label + str(i) +  ".png"
    img = plt.imread(fname)    
    Y_train[i,:,:,0] = img
    
    
    
#%% training the model
    
model.fit(X_train,Y_train, epochs = 5, batch_size = 128)

#%%  test data set
X_test = np.zeros((1,512,512,1))
X_test[0,:,:,0] = plt.imread("/Users/rockgu/Documents/U-net/unet/data/membrane/test/0.png")

check = model.predict(X_test)

plt.imshow(check[0,:,:,0])







