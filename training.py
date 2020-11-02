#training file

import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

(x, y), (x_test, y_test) = mnist.load_data()

x = x.astype('float32')
x_test = x_test.astype('float32')
x /= 255
x_test /= 255

print(x[0].shape)
print(x_test[0].shape)

x = x.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


print(x)
print(x_test)
y = to_categorical(y, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu',
    input_shape=x.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(x[:20000], y[:20000],
    validation_data=(x_test, y_test), epochs=3)

model.save("model.h5")

