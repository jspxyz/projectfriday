import librosa
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from test_recorder2 import record

# load the model
# code from predict_load_model
from models import model_a_conv1d

# assigning the pickle files
# with open('./Data_Array_Storage/X_train.pkl', 'rb') as f:
#     X_train = pickle.load(f)

print('Opening pickle files...')
with open('./Data_Array_Storage/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

# with open('./Data_Array_Storage/y_train.pkl', 'rb') as f:
#     y_train = pickle.load(f)

with open('./Data_Array_Storage/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

print('Loading model...')
# loading model with just h5
# Recreate the exact same model, including its weights and the optimizer
loaded_model = tf.keras.models.load_model('./models_saved/model_a_conv1d.h5')

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
                              res_type='kaiser_fast',
                              duration=2.5,
                              sr=44100,
                              offset=0.5
                             )

sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
newdf = pd.DataFrame(data=mfccs).T

# apply prediction
newdf= np.expand_dims(newdf, axis=2)
newpred = loaded_model.predict(newdf, 
                         batch_size=16, 
                         verbose=1)

# output prediction
# filename = '/content/labels'
# infile = open(filename,'rb')
# lb = pickle.load(infile)
# infile.close()
with open('./Data_Array_Storage/labels.pkl', 'rb') as f:
    lb = pickle.load(f)

# Get the final predicted label
final = newpred.argmax(axis=1)
final = final.astype(int).flatten()
final = (lb.inverse_transform((final)))
print(final) 