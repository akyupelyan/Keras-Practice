
#testing file

import sys
from keras.models import load_model
from keras.datasets import mnist
from numpy import loadtxt
from keras.datasets import mnist
from keras.utils import to_categorical


(x, y), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype('float32')

x_test /= 255


#print(x_test[0].shape)

x_test = x_test.reshape(10000, 28, 28, 1)

#print(x_test)

y_test = to_categorical(y_test, 10)


def main(arg):
    print("Starting test.py ... ")
    #print(arg)

    model = load_model(file)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("accuracy:", (scores[1]*100))

file = sys.argv[1]
main(file)