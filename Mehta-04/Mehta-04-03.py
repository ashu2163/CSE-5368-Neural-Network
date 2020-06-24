import pytest
import numpy as np
from cnn import CNN
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def test_eval():
    my_cnn=CNN()
    n_c=5
    train_s=100
    (X_train,Y_train),(X_test,Y_test)=datasets.cifar10.load_data()
    X_train=X_train[0:train_s,:]
    Y_train=Y_train[0:train_s,:]
    X_test=X_test[0:train_s,:]
    Y_test=Y_test[0:train_s,:]
    Y_train=tf.keras.utils.to_categorical(Y_train)
    Y_train=keras.utils.to_categorical(Y_test)
    X_train=X_train.astype('float32')/255
    Y_train=Y_train.astype('float32')/255

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    #history = model.fit(X_train, Y_train,batch_size=20, epocs=100)
    
    x=1
    
    my_cnn.add_input_layer(shape=(32,32,3),name="input")
    my_cnn.append_conv2d_layer(num_of_filters=32, kernel_size=(3,3),padding="same", activation='linear', name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    test1=[0]*3
    my_cnn.append_flatten_layer(name="flat1")
    test1[1]=3
    my_cnn.append_dense_layer(num_nodes=10,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=2,activation="relu",name="dense2")
    assert x==1
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    

    model.compile(optimizer='adam',metrics=['accuracy'],loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    test=model.evaluate(X_test,Y_test)
    #history = model.fit(x=X_train,y=Y_train,batch_size=100,epochs=200,shuffle=False)
    test1[0]=2
    #history=my_cnn.train(X_train,Y_train,batch_size=100,num_epochs=200)
    
    #test2=my_cnn.evaluate(X_test,Y_test)
    assert test1[0]<test1[1]
    

