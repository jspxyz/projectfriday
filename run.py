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
from flask import logging, Flask, render_template, request, jsonify
import datetime
import os
from os.path import join, dirname

# prediction import libraries
import librosa
import numpy as np
import pickle
import tensorflow as tf

# Watson API libraries
from ibm_watson import SpeechToTextV1
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions

# changing system path to import config modeul
import os, sys
sys.path.insert(1, os.getcwd())
from config import config_s2t_api, config_s2t_api_url, config_nlu_api, config_nlu_api_url

s2t_api = config_s2t_api
s2t_api_url = config_s2t_api_url
nlu_api = config_nlu_api
nlu_api_url = config_nlu_api_url

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

# recorder audio section
@app.route('/recorder', methods=['POST'])
def audio():
    basename = "audio.wav"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = "_".join([timestamp, basename]) # e.g. '20201207_1714_audio'
    savepath = './entries/audio'
    audio_filepath = "/".join([savepath, filename])
    with open(audio_filepath, 'wb') as f:
        f.write(request.data)
    # with open('./entries/audio.wav', 'wb') as f:
    #     f.write(request.data)
    proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', audio_filepath], text=True, stderr=PIPE)

    # predictions = get_audio_sentiment(audio_filepath)
    return proc.stderr #, audio_filepath

# def preprocess_audio_sentiment(audio_filepath):
#     entry = audio_filepath

#     # Transform the file to apply prediction
#     X, sample_rate = librosa.load(entry,
#                                     res_type='kaiser_best',
#                                     duration=2.5,
#                                     sr=44100,
#                                     offset=0.5
#                                     )
    
#     mfccs = librosa.feature.mfcc(y=X,
#                                 sr = sample_rate,
#                                 n_mfcc = 40)
    
#     # changing axis to be (time, features)
#     mfccs = np.moveaxis(mfccs, 0, -1)
#     mfccs = np.expand_dims(mfccs, axis=0)
    
#     return mfccs

# @app.route('/predict/', methods=['POST'])
# def get_audio_sentiment(audio_filepath):
#     mfccs = preprocess_audio_sentiment(audio_filepath)
    
#     probability = model.predict(mfccs) # add batch_size and verbose?

#     with open('./Data_Array_Storage/labels.pkl', 'rb') as f:
#         lb = pickle.load(f)

#     classes = lb.classes_

#     prob_index = probability.argmax(axis=1)
#     # prediction = final.astype(int).flatten() # doesn't seem like i need this
#     prediction = lb.inverse_transform((prob_index))
#     # prediction = classes[prob_index] # does the same as above:

#     # sending probability to list
#     prob_list = probability[0].tolist()

#     # creating class_prob dictionary
#     class_prob = [(classes[i], prob_list[i]) for i in range(len(classes))]

#     return jsonify({'label': prediction, 'probability': class_prob})

# ## Watson API section ##

# # working function taken from w_test_s2t_request.py file
# # Watson API function to transcribe speech to text
# # added on 20201214 1309
# def transcribe_audio(audio_filepath):

#     s2t_authenticator = IAMAuthenticator(s2t_api)
#     speech_to_text = SpeechToTextV1(
#         authenticator=s2t_authenticator
#     )

#     speech_to_text.set_service_url(s2t_api_url)

#     # removed dirname(__file__) at beginning of join()
#     with open(join('./.', audio_filepath),
#                 'rb') as audio_file:
#         s2t_results = speech_to_text.recognize(
#             audio=audio_file,
#             content_type='audio/wav'
#             # word_alternatives_threshold=0.9,
#             # keywords=['colorado', 'tornado', 'tornadoes'],
#             # keywords_threshold=0.5
#         ).get_result()
#     # print(type(speech_recognition_results))
#     # found that speech_recognition_results is a dictionary
#     # removing the below because json.dumps converts into a string
#     # transcribe_audio_result = json.dumps(speech_recognition_results, indent=2)
#     return s2t_results

# # code taken from w_test_nlu_keywords.py
# # Watson API function to get text sentiment
# def get_text_sentiment(text):
#     nlu_authenticator = IAMAuthenticator(nlu_api)
#     natural_language_understanding = NaturalLanguageUnderstandingV1(
#         version='2020-08-01',
#         authenticator=nlu_authenticator
#     )

#     natural_language_understanding.set_service_url(nlu_api_url)

#     # text_to_analyze = text

#     # with open(text_to_analyze) as f:
#     #     contents = f.readlines()

#     nlu_response = natural_language_understanding.analyze(
#         # url='www.ibm.com',
#         # text=' '.join(contents),
#         text=text,
#         features=Features(keywords=KeywordsOptions(sentiment=True,emotion=True,limit=2))).get_result()

#     # print(json.dumps(response, indent=2))
#     return nlu_response

if __name__ == "__main__":
    app.logger = logging.getLogger() # 'audio-gui'
    app.run()


# prediction code
# Get the final predicted label
# final = newpred.argmax(axis=1)
# final = final.astype(int).flatten()
# final = (lb.inverse_transform((final)))
# print(final) 

# code from vnd classifier def predict()
    # probs = model.predict(image)
    # label = np.argmax(probs, axis=1)
    # label = class_names[label[0]]
    # probs = probs[0].tolist()
    # probs = [(probs[i], class_names[i]) for i in range(len(class_names))]
 
    # return jsonify({'label': label, 'probs': probs}) 