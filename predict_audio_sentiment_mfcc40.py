import librosa
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from recorder2 import record

# load the model
# code from predict_load_model
from models import model_a_conv1d

# assigning the pickle files
# with open('./Data_Array_Storage/X_train.pkl', 'rb') as f:
#     X_train = pickle.load(f)

print('Opening pickle files...')
with open('./Data_Array_Storage/X_test_mfcc40_pol_0dn_us.pkl', 'rb') as f:
    X_test = pickle.load(f)

# with open('./Data_Array_Storage/y_train.pkl', 'rb') as f:
#     y_train = pickle.load(f)

with open('./Data_Array_Storage/y_test_mfcc40_pol_0dn_us.pkl', 'rb') as f:
    y_test = pickle.load(f)

print('Loading model...')
# loading model with just h5
# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('./models_saved/model_d_conv1d_mfcc40_0dn_us_pol_b32.h5')

print('Loaded model. Spinning up the recorder...')
# recorder for X seconds
# returns the wav file path
prediction_sound = record(5)

print('This is what you recorded!')
os.system("afplay " + prediction_sound)

print('Making prediction...')
# convert into librosa
data, sampling_rate = librosa.load(prediction_sound)

# Transform the file so we can apply the predictions
X, sample_rate = librosa.load(prediction_sound,
                              res_type='kaiser_best',
                              duration=2.5,
                              sr=44100,
                              offset=0.5
                             )


mfccs = librosa.feature.mfcc(y=X, 
                            sr=sample_rate,
                            n_mfcc=40)

mfccs = np.moveaxis(mfccs, 0, -1)

print('mfccs looks like: ', mfccs)
print('mfccs shape is: ', mfccs.shape)
# newdf = pd.DataFrame(data=mfccs).T

mfccs = np.expand_dims(mfccs, axis=0)
print('mfccs expanded shape is: ', mfccs.shape)

# apply prediction
# newdf= np.expand_dims(newdf, axis=2)
prediction = model.predict(mfccs)
                        #  batch_size=16, 
                        #  verbose=1)

print('prediction looks like: ', prediction)

# output prediction
# filename = '/content/labels'
# infile = open(filename,'rb')
# lb = pickle.load(infile)
# infile.close()
with open('./Data_Array_Storage/labels_mfcc40_pol_0dn_us.pkl', 'rb') as f:
    lb = pickle.load(f)

print('label pickle file is: ', lb)
# Get the final predicted label
final = prediction.argmax(axis=1)
print('prediction.argmax: ', final)

final = final.astype(int).flatten()
print('prediction flatten: ', final)

final = (lb.inverse_transform((final)))
print(final) 