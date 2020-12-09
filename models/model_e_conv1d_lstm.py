'''
New conv1d + LSTM model for mfcc40 features
Created on December 9, 2020 at 1100
https://github.com/vandana-rajan/1D-Speech-Emotion-Recognition/blob/master/cnn1d.py
'''
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D, LSTM
from tensorflow.keras.layers import AveragePooling1D, GlobalAveragePooling2D, MaxPooling1D
from tensorflow.keras.models import Model, model_from_json, Sequential

def conv1d_lstm(input_shape):
    
    learning_rate = 0.0001
    decay = 1e-6
    momentum = 0.9
    num_classes=3
    num_fc = 64
    
    model = Sequential(name='conv1d_lstm')
    
    # LFLB1
    model.add(Conv1D(filters = 64,kernel_size = (3),strides=1,padding='same',data_format='channels_last',input_shape=input_shape))	
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size = 4, strides = 4))

    #LFLB2
    model.add(Conv1D(filters=64, kernel_size = 3, strides=1,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size = 4, strides = 4))

    #LFLB3
    model.add(Conv1D(filters=128, kernel_size = 3, strides=1,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size = 4, strides = 4))

    #LFLB4
    model.add(Conv1D(filters=128, kernel_size = 3, strides=1,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size = 4, strides = 4))

    #LSTM
    model.add(LSTM(units=num_fc)) 

    #FC
    model.add(Dense(units=num_classes,activation='softmax'))

    #Model compilation	
    opt = tf.keras.optimizers.SGD(lr = learning_rate, decay=decay, momentum=momentum, nesterov=True)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    return model