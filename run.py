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

from predict_audio_sentiment_outputdict_clean import *

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

# empty dictionary to save results
results_dict = {}

filepath = './entries/audio/test_r2_testing.wav'

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
    result = proc.stderr
    result = '\n'.join(result.split("\n")[-4:])
    # predictions = get_audio_sentiment(audio_filepath)

    # adding date to dictionary
    results_dict['date'] = timestamp.split('_')[0]

    filepath = audio_filepath

    print('The audio filepath is: ', filepath)
    audio = filepath

    print('This is what you recorded!')
    os.system("afplay " + audio)

    print("Transcribing audio....\n")
    # result = transcribe_audio('speech.wav')
    s2t_results = transcribe_audio(audio)


    text = s2t_results['results'][0]['alternatives'][0]['transcript']

    print("Analyzing text sentiment...\n")
    text_sentiment_results = get_text_sentiment(text)

        # saving text results to dictionary
    text = s2t_results['results'][0]['alternatives'][0]['transcript']
    results_dict['text'] = text
    results_dict['text_confidence'] = s2t_results['results'][0]['alternatives'][0]['confidence']

    # saving text sentiment results to dictionary
    results_dict['text_wordcount'] = len(text.split()) 
    results_dict['keywords'] = text_sentiment_results['keywords'][0]['text']
    results_dict['text_polarity'] = text_sentiment_results['keywords'][0]['sentiment']['label']
    results_dict['text_polarity_prob'] = text_sentiment_results['keywords'][0]['sentiment']
    emotion_dict = text_sentiment_results['keywords'][0]['emotion']
    emotion_max = max(emotion_dict, key=emotion_dict.get)
    results_dict['text_emotion'] = emotion_max
    results_dict['text_emotion_prob'] = text_sentiment_results['keywords'][0]['emotion']

    # saving audio sentiment to dictionary
    print('audio sentiment analysis: ')
    with open('./Data_Array_Storage/labels_mfcc40_pol_0dn_us.pkl', 'rb') as f:
        lb = pickle.load(f)

    classes = lb.classes_

    audio_pol_probability = get_audio_sentiment_pol(audio)

    # Get the final predicted label
    audio_pol_prob_index = audio_pol_probability.argmax(axis=1) # this outputs the highest index - example: [1]


    audio_pol_prediction = lb.inverse_transform((audio_pol_prob_index))
    audio_pol_prob_list = audio_pol_probability[0].tolist()

    audio_pol_class_prob = [(classes[i], audio_pol_prob_list[i]) for i in range(len(classes))]

    results_dict['audio_polarity'] = audio_pol_prediction[0]
    results_dict['audio_polarity_prob'] = dict(audio_pol_class_prob)

    print('this is the final dictionary output')
    print(results_dict)

    return jsonify([result, results_dict]) #, audio_filepath
    

if __name__ == "__main__":
    app.logger = logging.getLogger() # 'audio-gui'
    app.run()
