import tensorflow as tf

model = tf.keras.models.load_model(path to h5)

model.summary()

# freeze first 2 layers
# reuse the last 4

# base_model is the loaded model
base_model.summary()

for layer in base_model.layers[:2]:
    layer.trainable = False

# output of 3rd layer to add in more layers
# grabbing the last 4 layers and using as inputs
base_model_test = tf.keras.models.Model(base_model.input, base_model.get_layer("flatten_1").output)

base_model_test = sequential(
                        tf.keras.layers.Dense(3))

base_model_test.summary()