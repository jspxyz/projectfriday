import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling1D, GlobalAveragePooling2D, MaxPooling1D
from tensorflow.keras.models import Model, model_from_json, Sequential

def model_a_conv2d():
    # callback list: ModelCheckpoint, reduceLROnPlat, EarlyStopping
    checkpoint_path = "model_conv2d.h5"

    # Create a callback that saves the model's weights
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=False,
                                                    save_best_only=True,
                                                    verbose=1),
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                    patience=5, 
                                                    restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                        patience=2, 
                                                        factor=0.5, 
                                                        min_lr=0.00001, 
                                                        verbose=1)]
    
    # New model
    model = Sequential()
    model.add(Conv2D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))  # X_train.shape[1] = No. of Columns
    model.add(Activation('relu'))
    model.add(Conv2D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv2D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv2D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(6)) # Target class number
    model.add(Activation('softmax'))
    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    model.summary()