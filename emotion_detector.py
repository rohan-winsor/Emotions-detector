#!/usr/bin/env python

import keras
from keras import layers, Model, optimizers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Activation
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_data():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    classes = np.array(test_dataset["list_classes"][:])
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    y_train = train_set_y_orig.T
    y_test = test_set_y_orig.T
    x_train = train_set_x_orig/255
    x_test = test_set_x_orig/255
    return x_train, x_test, y_train, y_test


def model_creation(train):
    X = layers.Input(train)
    Con2D1 = Conv2D(16, (3, 3))(X)
    ACon2D1 = Activation('relu')(Con2D1)
    MaxPool1 = MaxPool2D()(ACon2D1)
    BatchNorm = BatchNormalization()(MaxPool1)
    Con2D2 = Conv2D(32, (3, 3))(BatchNorm)
    MaxPool2 = MaxPool2D()(Con2D2)
    flatten = Flatten()(MaxPool2)
    FinalLayer = Dense(1, activation='sigmoid')(flatten)
    model = Model(input=X, output=FinalLayer, name='model')
    return model


def model_training(model, x_train, y_train, epochs=40):
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs)
    return model, history


def model_eval(model, x_test, y_test):
    score, acc = model.evaluate(x=x_test,
                                y=y_test)
    return score, acc


def model_check(model, file_name):
    file_name = 'datasets\\random_test\\' + file_name
    img = image.load_img(file_name, target_size=(64, 64))
    img_org = image.load_img(file_name)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    pred = {1:"\nHappy\n",0:'\nNot Happy\n'}
    plt.imshow(img_org)
    plt.title(pred[int(y[0][0])])
    plt.show()


if __name__ == '__main__':
    Model = False
    if Model:
        x_train, x_test, y_train, y_test = load_data()
        model = model_creation(x_train.shape[1:])
        model, history = model_training(model, x_train, y_train, epochs=40)
        model.save("my_model.h5")
        model_eval(model, x_test, y_test)
    else:
        model = load_model('my_model.h5')
    model_check(model, file_name='test.jpg')
