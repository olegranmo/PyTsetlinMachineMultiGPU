from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import cifar10

import cv2

factor = 20
clauses = int(4000*factor)
T = int(75*10*factor)
s = 20.0
patch_size = 8

labels = [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

Y_test=Y_test.reshape(Y_test.shape[0])
for i in range(X_test.shape[0]):
        for j in range(X_test.shape[3]):
                X_test[i,:,:,j] = cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

datagen = ImageDataGenerator(
    rotation_range=0,
    horizontal_flip=False,
    width_shift_range=0,
    height_shift_range=0
    #zoom_range=0.3
    )
datagen.fit(X_train)

# Introduce augmented data here

f = open("cifar10_%.1f_%d_%d_%d.txt" % (s, clauses, T,  patch_size), "w+")

tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (patch_size, patch_size), number_of_gpus=16)

batch = 0
for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=10000):
        batch += 1
        Y_batch = Y_batch.reshape(Y_batch.shape[0]).astype(np.int32)
        X_batch = X_batch.reshape(X_batch.shape[0], 32, 32, 3).astype(np.uint8)

        for i in range(X_batch.shape[0]):
                for j in range(X_batch.shape[3]):
                        X_batch[i,:,:,j] = cv2.adaptiveThreshold(X_batch[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

                print(X_batch[i,:,:,:].shape)


        start_training = time()
        tm.fit(X_batch, Y_batch, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result_test = 100*(tm.predict(X_test) == Y_test).mean()
        stop_testing = time()

        result_train = 100*(tm.predict(X_batch) == Y_batch).mean()
        print("%d %.2f %.2f %.2f %.2f" % (batch, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
        print("%d %.2f %.2f %.2f %.2f" % (batch, result_train, result_test, stop_training-start_training, stop_testing-start_testing), file=f)
        f.flush()
f.close()
