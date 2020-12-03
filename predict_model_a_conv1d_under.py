import pickle
import tensorflow as tf

from models import model_a_conv1d

# assigning the pickle files
# with open('./Data_Array_Storage/X_train.pkl', 'rb') as f:
#     X_train = pickle.load(f)

with open('./Data_Array_Storage/X_test_under.pkl', 'rb') as f:
    X_test_under = pickle.load(f)

# with open('./Data_Array_Storage/y_train.pkl', 'rb') as f:
#     y_train = pickle.load(f)

with open('./Data_Array_Storage/y_test_under.pkl', 'rb') as f:
    y_test_under = pickle.load(f)

# loading model with just h5
# Recreate the exact same model, including its weights and the optimizer
model_from_h5 = tf.keras.models.load_model('./models_saved/model_a_conv1d_under.h5')

# Show the model architecture
model_from_h5.summary()

# We need to define its optimizer and loss function again since the h5 file
# only need to compile if continuing training
# if just predictions, then no need for compile
# does not contain those information :(
# model_from_h5.compile(optimizer='Adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# Re-evaluate the model
loss, acc = model_from_h5.evaluate(X_test_under, y_test_under)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))