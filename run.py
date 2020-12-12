# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
# flask import libraries
from flask_migrate import Migrate
from os import environ
from sys import exit
from decouple import config

from config_flask import config_dict
from app import create_app, db

# recorder input libraries
from subprocess import run, PIPE
from flask import logging, Flask, render_template, request
import datetime
import os

# prediction import libraries
import librosa
import numpy as np
import tensorflow as tf

# WARNING: Don't run with debug turned on in production!
DEBUG = config('DEBUG', default=True)

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:
    
    # Load the configuration using the default values 
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app( app_config ) 
Migrate(app, db)

model = tf.keras.models.load_model('./models_saved/model_d_conv1d_mfcc40_0dn_us_pol_b32.h5')

@app.route('/recorder', methods=['POST'])
def audio():
    basename = "audio.wav"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = "_".join([timestamp, basename]) # e.g. '20201207_1714_audio'
    savepath = './entries/audio'
    filepath = "/".join([savepath, filename])
    with open(filepath, 'wb') as f:
        f.write(request.data)
    # with open('./entries/audio.wav', 'wb') as f:
    #     f.write(request.data)
    proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', filepath], text=True, stderr=PIPE)
    return proc.stderr, filepath

def preprocess_audio_sentiment(filepath):
    entry = filepath

    # Transform the file to apply prediction
    X, sample_rate = librosa.load(entry,
                                    res_type='kaiser_best',
                                    duration=2.5,
                                    sr=44100,
                                    offset=0.5
                                    )
    
    mfccs = librosa.feature.mfcc(y=X,
                                sr = sample_rate,
                                n_mfcc = 40)
    
    # changing axis to be (time, features)
    mfccs = np.moveaxis(mfccs, 0, -1)
    mfccs = np.expand_dims(mfccs, axis=0)
    
    return mfccs

@app.route('/predict/', methods=['POST'])
def predict_audio_sentiment(mfccs):
    probability = model.predict(mfccs) # add batch_size and verbose?

    with open('./Data_Array_Storage/labels.pkl', 'rb') as f:
        labels = pickle.load(f)

    prediction = probability
    prediction = final.astype(int).flatten()
    prediction = (labels.inverse_transform((predition)))


    label = np.argmax(probability, axis=1)

    

# output prediction
filename = '/content/labels'
infile = open(filename,'rb')
lb = pickle.load(infile)
infile.close()

# Get the final predicted label
final = newpred.argmax(axis=1)
final = final.astype(int).flatten()
final = (lb.inverse_transform((final)))
print(final) 

if __name__ == "__main__":
    app.logger = logging.getLogger() # 'audio-gui'
    app.run()
