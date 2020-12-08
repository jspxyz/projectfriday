import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import AveragePooling1D, GlobalAveragePooling2D, MaxPooling1D, MaxPool2D
from tensorflow.keras.models import Model, model_from_json, Sequential

def model_c_conv2d(input_shape):
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation = 'relu', strides = (1,1), padding = 'same', input_shape = input_shape))
    model.add(Conv2D(32, (3,3), activation = 'relu', strides = (1,1), padding = 'same'))
    model.add(Conv2D(64, (3,3), activation = 'relu', strides = (1,1), padding = 'same'))
    model.add(Conv2D(128, (3,3), activation = 'relu', strides = (1,1), padding = 'same'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation ='relu'))
    model.add(Dense(64, activation ='relu'))
    model.add(Dense(3, activation ='softmax'))
    model.summary()
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    
    return model