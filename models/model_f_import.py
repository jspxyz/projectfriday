import tensorflow as tf

from tensorflow.keras.models import Model, Sequential

model_loaded = tf.keras.models.load_model('./models_saved/Emotion_Voice_Detection_Model.h5')

model_loaded.summary()

# freeze first 2 layers
# reuse the last 4

# model_loaded is the loaded model
model_loaded.summary()

# loop to freeze first 2 layers
for layer in model_loaded.layers[:2]:
    layer.trainable = False

# output of 3rd layer to add in more layers
# grabbing the last 4 layers and using as inputs
model_altered = tf.keras.models.Model(model_loaded.input, model_loaded.get_layer("flatten_1").output)

model_altered = Sequential(tf.keras.layers.Dense(3))

model_altered.summary()