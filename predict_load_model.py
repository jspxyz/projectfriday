import tensorflow as tf

# loading model with just h5
# Recreate the exact same model, including its weights and the optimizer
model_from_h5 = tf.keras.models.load_model('./models_h5/model_a_conv1d.h5')

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
loss, acc = model_from_h5.evaluate(X_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))