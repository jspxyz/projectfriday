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

# clean up imports later
from predict_audio_sentiment_outputdict_clean import *
from db_entries import *

# database stuff
import sqlite3

# changing system path to import config modeul
import os, sys
sys.path.insert(1, os.getcwd())
from config import config_s2t_api, config_s2t_api_url, config_nlu_api, config_nlu_api_url

s2t_api = config_s2t_api
s2t_api_url = config_s2t_api_url
nlu_api = config_nlu_api
nlu_api_url = config_nlu_api_url

# audio variable settings
res_type = 'kaiser_best'
duration = 5
sr = 44100
offset = 0.1
n_mfcc = 40
max_len = round(duration * sr / 512)

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

# model = tf.keras.models.load_model('./models_saved/model_d_conv1d_mfcc40_0dn_us_pol_b32.h5')

# empty dictionary to save results
results_dict = {}

filepath = './entries/audio/test_r2_testing.wav'

# from datetime import datetime, timezone

# def utc_to_local(utc_dt):
#     return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)

# recorder audio section
@app.route('/recorder', methods=['POST'])
def audio():
    basename = "audio.wav"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

    # saving audio filepath
    results_dict['entry_filepath'] = audio_filepath

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
    results_dict['text_content'] = text
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

    # saving audio sentiment polarity to dictionary
    print('audio sentiment polarity analysis: ')
    # with open('./Data_Array_Storage/pol_duration5_axis0_us_labels.pkl', 'rb') as f:
    with open('./Data_Array_Storage/male_pol_us_labels.pkl', 'rb') as f:
        audio_pol_lb = pickle.load(f)

    audio_pol_classes = audio_pol_lb.classes_

    audio_pol_probability = get_audio_sentiment_pol(audio, res_type, duration, sr, offset, n_mfcc, max_len)

    # Get the final predicted label
    audio_pol_prob_index = audio_pol_probability.argmax(axis=1) # this outputs the highest index - example: [1]


    audio_pol_prediction = audio_pol_lb.inverse_transform((audio_pol_prob_index))
    audio_pol_prob_list = audio_pol_probability[0].tolist()

    audio_pol_class_prob = [(audio_pol_classes[i], audio_pol_prob_list[i]) for i in range(len(audio_pol_classes))]

    results_dict['audio_polarity'] = audio_pol_prediction[0]
    results_dict['audio_polarity_prob'] = dict(audio_pol_class_prob)

    ####### start audio sentiment emotion section ######
    # saving audio sentiment emotion to dictionary
    print('audio sentiment emotion analysis: ')
    # with open('./Data_Array_Storage/emo_duration5_axis0_labels.pkl', 'rb') as f:
    with open('./Data_Array_Storage/male_emo_labels.pkl', 'rb') as f:
        audio_emo_lb = pickle.load(f)

    audio_emo_classes = audio_emo_lb.classes_

    audio_emo_probability = get_audio_sentiment_emo(audio, res_type, duration, sr, offset, n_mfcc, max_len)

    audio_emo_prob_index = audio_emo_probability.argmax(axis=1) # this outputs the highest index - example: [1]

    audio_emo_prediction = audio_emo_lb.inverse_transform((audio_emo_prob_index))

    audio_emo_prob_list = audio_emo_probability[0].tolist()

    audio_emo_class_prob = [(audio_emo_classes[i], audio_emo_prob_list[i]) for i in range(len(audio_emo_classes))]

    results_dict['audio_emotion'] = audio_emo_prediction[0]
    results_dict['audio_emotion_prob'] = dict(audio_emo_class_prob)
    ###### end audio sentiment polarity section

    print('this is the final dictionary output')
    print(results_dict)

    # connecting to database
    conn = sqlite3.connect('journal.db')

    # create a cursor
    cur = conn.cursor()

    create_journal_entries_table(cur)
    conn.commit()

    entry = Journal_Entry(results_dict, cur)
    entry.save_into_db()

    conn.commit()
    conn.close()

    results_dict.pop('date')
    results_dict.pop('entry_filepath')

    # change prediction table here


    return jsonify([result, results_dict]) #, audio_filepath
    

if __name__ == "__main__":
    app.logger = logging.getLogger() # 'audio-gui'
    app.run()
