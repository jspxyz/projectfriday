import numpy as np
import pickle
import tensorflow as tf

from models import model_a_conv1d

# assigning the pickle files
with open('./Data_Array_Storage/X_train_smote.pkl', 'rb') as f:
    X_train_smote = pickle.load(f)

with open('./Data_Array_Storage/X_test_smote.pkl', 'rb') as f:
    X_test_smote = pickle.load(f)

with open('./Data_Array_Storage/y_train_smote.pkl', 'rb') as f:
    y_train_smote = pickle.load(f)

with open('./Data_Array_Storage/y_test_smote.pkl', 'rb') as f:
    y_test_smote = pickle.load(f)


model = model_a_conv1d()
optimizer = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)

# callback list: ModelCheckpoint, reduceLROnPlat, EarlyStopping
checkpoint_path = "./models_saved/model_a_conv1d_smote.h5"

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


model_history=model.fit(X_train_smote, 
                        y_train_smote,
                        batch_size=16,
                        epochs=150,
                        validation_data=(X_test_smote, y_test_smote),
                        verbose=2,
                        callbacks=callbacks)

# testing when numpy could not convert to tensor
# with open('./Data_Array_Storage/X_train_smote.npy', 'rb') as f:
#     X_train_smote = np.load(f, allow_pickle=True)

# print(type(X_train_smote))
# X_train_smote.shape()

# X_train_tensor = tf.convert_to_tensor(X_train_smote)
# print(type(X_train_tensor))

# # numpy arrays
# with open('test.npy', 'wb') as f:
#     np.save(f, np.array([1, 2]))
#     np.save(f, np.array([1, 3]))
# with open('test.npy', 'rb') as f:
#     a = np.load(f)
#     b = np.load(f)