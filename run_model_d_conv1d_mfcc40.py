'''
Creating to run Model D Conv1d_mfcc40
different layer system
full mfcc features
Created on 2020.12.08 at 1843

'''
import numpy as np
import pickle
import tensorflow as tf

from models import model_d_conv1d

# assigning the pickle files
with open('./Data_Array_Storage/X_train_mfcc40_axis0.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('./Data_Array_Storage/X_test_mfcc40_axis0.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('./Data_Array_Storage/y_train_mfcc40_axis0.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('./Data_Array_Storage/y_test_mfcc40_axis0.pkl', 'rb') as f:
    y_test = pickle.load(f)

input_shape = (X_train.shape[1], X_train.shape[2])

model = model_d_conv1d(input_shape)
optimizer = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)

# callback list: ModelCheckpoint, reduceLROnPlat, EarlyStopping
checkpoint_path = "./models_saved/model_d_conv1d.h5"

# Create a callback that saves the model's weights
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 save_best_only=True,
                                                 verbose=1), # 1 tells your which epoch is saving
               tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                patience=5, 
                                                restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                    patience=2, 
                                                    factor=0.5, 
                                                    min_lr=0.00001, 
                                                    verbose=1)]

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


model_history=model.fit(X_train, 
                        y_train,
                        batch_size=16,
                        epochs=150,
                        validation_data=(X_test, y_test),
                        verbose=2,
                        callbacks=callbacks)

