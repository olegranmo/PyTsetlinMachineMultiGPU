# Tsetlin Machine Multi-GPU

### CIFAR Demo

#### Code: CIFARDemo.py

```python
from PyTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import cifar10

import cv2

factor = 80
clauses = int(4000*factor)
T = int(75*10*factor)
s = 20.0
patch_size = 8

labels = [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck']

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

Y_test=Y_test.reshape(Y_test.shape[0])
for i in range(X_test.shape[0]):
        for j in range(X_test.shape[3]):
                X_test[i,:,:,j] = cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)#cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
    #zoom_range=0.3
    )
datagen.fit(X_train)

f = open("cifar10_%.1f_%d_%d_%d.txt" % (s, clauses, T,  patch_size), "w+")

tm = MultiClassConvolutionalTsetlinMachine2D(clauses, T, s, (patch_size, patch_size), number_of_gpus=16)

batch = 0
for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=10000):
        batch += 1
        
        Y_batch = Y_batch.reshape(Y_batch.shape[0]).astype(np.int32)
        X_batch = X_batch.reshape(X_batch.shape[0], 32, 32, 3).astype(np.uint8)

        for i in range(X_batch.shape[0]):
                for j in range(X_batch.shape[3]):
                        X_batch[i,:,:,j] = cv2.adaptiveThreshold(X_batch[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 

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
```

#### Output

```bash
python ./CIFARDemo.py

Preparing GPU #0
Preparing GPU #1
Preparing GPU #2
Preparing GPU #3
Preparing GPU #4
Preparing GPU #5
Preparing GPU #6
Preparing GPU #7
Preparing GPU #8
Preparing GPU #9
Preparing GPU #10
Preparing GPU #11
Preparing GPU #12
Preparing GPU #13
Preparing GPU #14
Preparing GPU #15

1 45.11 41.06 26.91 11.10
2 52.29 47.12 21.88 10.81
3 57.00 49.78 21.81 10.96
...

278 99.98 70.30 20.64 10.94
279 99.97 70.20 20.73 10.89
280 99.97 70.33 20.72 10.86
```
