import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import h5py
import numpy as np
#x_train_set, y_train_set, x_test_set, y_test_set, classes= load_dataset()
train_images='face_train.h5'
test_images='face_test.h5'
train_dataset = h5py.File(train_images, "r")
train_data = np.array(train_dataset["train_data"])
train_label = np.array(train_dataset["train_label"])
test_dataset = h5py.File(test_images, "r")
test_data = np.array(test_dataset["test_data"])
test_label = np.array(test_dataset["test_label"])
print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)
train_data=np.double(train_data)/255
test_data=np.double(test_data)/255
print(type(train_data))
print(type(test_data))
print(train_data[0][0])
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(112, 92, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(40, activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
history = model.fit(train_data, train_label, epochs=20, 
                    validation_data=(test_data, test_label))
