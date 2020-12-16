'''
This file is for Watson API and audio sentiment
This is the same as predict_audio_sentiment_combined
    But outputs a dictionary of the information
    CLEAN VERSION
Uses Recorder2 to save audio
Uses s2t and nlu api to get text, confidence of text, and sentiment.
Uses Audio MFCC40 to get audio sentiment.
Created on 20201214 1452
'''

# libraries for audio sentiment
import librosa
import numpy as np
import pickle
import tensorflow as tf

import os
import json
from os.path import join, dirname
# from dotenv import load_dotenv
# from watson_developer_cloud import SpeechToTextV1 as SpeechToText
# from watson_developer_cloud import AlchemyLanguageV1 as AlchemyLanguage

from ibm_watson import SpeechToTextV1
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions

# from speech_sentiment_python.recorder import Recorder
from recorder2 import record

# changing system path to import config modeul
import os, sys
sys.path.insert(1, os.getcwd())
from config import config_s2t_api, config_s2t_api_url, config_nlu_api, config_nlu_api_url

s2t_api = config_s2t_api
s2t_api_url = config_s2t_api_url
nlu_api = config_nlu_api
nlu_api_url = config_nlu_api_url

# empty dictionary to save results
results_dict = {}

filepath = './entries/audio/test_r2_testing.wav'

# working function taken from w_test_s2t_request.py file
# Watson API function to transcribe speech to text
def transcribe_audio(path_to_audio_file):

    s2t_authenticator = IAMAuthenticator(s2t_api)
    speech_to_text = SpeechToTextV1(
        authenticator=s2t_authenticator
    )

    speech_to_text.set_service_url(s2t_api_url)

    # removed dirname(__file__) at beginning of join()
    with open(join('./.', path_to_audio_file),
                'rb') as audio_file:
        s2t_results = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav'
            # word_alternatives_threshold=0.9,
            # keywords=['colorado', 'tornado', 'tornadoes'],
            # keywords_threshold=0.5
        ).get_result()
    # print(type(s2t_results))
    # found that s2t_results is a dictionary
    # removing the below because json.dumps converts into a string
    # transcribe_audio_result = json.dumps(s2t_results, indent=2)

    return s2t_results

# code taken from w_test_nlu_keywords.py
# Watson API function to get text sentiment
def get_text_sentiment(text):
    nlu_authenticator = IAMAuthenticator(nlu_api)
    natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2020-08-01',
        authenticator=nlu_authenticator
    )

    natural_language_understanding.set_service_url(nlu_api_url)

    # text_to_analyze = text

    # with open(text_to_analyze) as f:
    #     contents = f.readlines()

    nlu_response = natural_language_understanding.analyze(
        # url='www.ibm.com',
        # text=' '.join(contents),
        text=text,
        features=Features(keywords=KeywordsOptions(sentiment=True,emotion=True,limit=5))).get_result()

    # print(json.dumps(response, indent=2))
    return nlu_response

# function to get audio sentiment
# code pulled from predict_audio_sentiment_mfcc40
def get_audio_sentiment_pol(path_to_audio_file, res_type, duration, sr, offset, n_mfcc):
    print('Loading audio sentiment polarity model...')
    # loading model with just h5
    # Recreate the exact same model, including its weights and the optimizer
    model = tf.keras.models.load_model('./models_saved/model_conv2d_axis0_dur5_pol_b16.h5')

    audio = path_to_audio_file

    print('Making prediction...')
    # convert into librosa
    # data, sampling_rate = librosa.load(audio)

    # Transform the file so we can apply the predictions
    X, sample_rate = librosa.load(audio,
                                res_type=res_type,
                                duration=duration,
                                sr=sr,
                                offset=offset
                                )


    mfccs = librosa.feature.mfcc(y=X, 
                                sr=sample_rate,
                                n_mfcc=n_mfcc)

    mfccs = np.moveaxis(mfccs, 0, -1)

    # print('mfccs looks like: ', mfccs)
    # print('mfccs shape is: ', mfccs.shape)
    # # newdf = pd.DataFrame(data=mfccs).T

    mfccs = np.expand_dims(mfccs, axis=0)
    # print('mfccs expanded shape is: ', mfccs.shape)

    # apply prediction
    # newdf= np.expand_dims(newdf, axis=2)
    audio_pol_probability = model.predict(mfccs)
                            #  batch_size=16, 
                            #  verbose=1)

    return audio_pol_probability

# function to get audio sentiment
# code pulled from predict_audio_sentiment_mfcc40
def get_audio_sentiment_emo(path_to_audio_file, res_type, duration, sr, offset, n_mfcc):
    print('Loading audio sentiment emotion model...')
    # loading model with just h5
    # Recreate the exact same model, including its weights and the optimizer
    model = tf.keras.models.load_model('./models_saved/model_conv2d_axis0_dur5_emo_b16.h5')

    audio = path_to_audio_file

    print('Making audio emotion prediction...')
    # convert into librosa
    # data, sampling_rate = librosa.load(audio)

    # Transform the file so we can apply the predictions
    X, sample_rate = librosa.load(audio,
                                res_type=res_type,
                                duration=duration,
                                sr=sr,
                                offset=offset
                                )


    mfccs = librosa.feature.mfcc(y=X, 
                                sr=sample_rate,
                                n_mfcc=n_mfcc)

    mfccs = np.moveaxis(mfccs, 0, -1)

    mfccs = np.expand_dims(mfccs, axis=0)

    audio_emo_probability = model.predict(mfccs)
                            #  batch_size=16, 
                            #  verbose=1)

    return audio_emo_probability

    
def main():
    # recorder = Recorder("speech.wav")
    # record function from test_recorder2.py
    # returns WAVE_OUTPUT_FILENAME
    print('The audio filepath is: ', filepath)
    audio = filepath

    print('This is what you recorded!')
    os.system("afplay " + audio)

    print("Transcribing audio....\n")
    # result = transcribe_audio('speech.wav')
    s2t_results = transcribe_audio(audio)

    # print('-'*10)
    # print('This is the Text Section')
    # assigning text
    text = s2t_results['results'][0]['alternatives'][0]['transcript']
    # text_confidence = s2t_results['results'][0]['alternatives'][0]['confidence']
    # print("Text: " + text + "\n")
    # print("Confidence: " + str(text_confidence) + "\n")
    
    # print('-'*10)
    print("Analyzing text sentiment...\n")
    # this is the text sentiment results
    text_sentiment_results = get_text_sentiment(text)
    # print('The text sentiment results are: ', text_sentiment_results)

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

    ###### saving audio sentiment polarity to dictionary #####
    print('audio sentiment polarity analysis: ')
    with open('./Data_Array_Storage/labels_mfcc40_pol_0dn_us.pkl', 'rb') as f:
        audio_pol_lb = pickle.load(f)

    audio_pol_classes = audio_pol_lb.classes_

    audio_pol_probability = get_audio_sentiment_pol(audio)

    # print('label audio_pol_classes: ', audio_pol_classes)
    # Get the final predicted label
    audio_pol_prob_index = audio_pol_probability.argmax(axis=1) # this outputs the highest index - example: [1]
    # print('audio_pol_probability.argmax: ', audio_pol_prob_index)

    # pred_label = audio_pol_classes[audio_pol_prob_index]
    # print('audio_pol_classes[audio_pol_prob_index]: ', pred_label)

    # final = final.astype(int).flatten()
    # print('prediction flatten: ', final)

    audio_pol_prediction = audio_pol_lb.inverse_transform((audio_pol_prob_index))
    # print('audio_pol_lb.inverse_transform of audio_pol_prob_index: ', prediction) 

    # print('trying to get out list of audio_pol_classes and its audio_pol_probability')
    # print('step 1 is to get probabilty list')
    audio_pol_prob_list = audio_pol_probability[0].tolist()
    # print('audio_pol_prob_list is: ', audio_pol_prob_list)

    audio_pol_class_prob = [(audio_pol_classes[i], audio_pol_prob_list[i]) for i in range(len(audio_pol_classes))]
    # print('audio_pol_classes and its audio_pol_probability: ', audio_pol_class_prob)

    # print({'label': audio_pol_prediction, 'audio_pol_probability': audio_pol_class_prob})

    results_dict['audio_polarity'] = audio_pol_prediction[0]
    results_dict['audio_polarity_prob'] = dict(audio_pol_class_prob)
    ####### end audio sentiment polarity section #######

    ####### start audio sentiment emotion section ######
    # saving audio sentiment emotion to dictionary
    print('audio sentiment emotion analysis: ')
    with open('./Data_Array_Storage/labels_mfcc40_emo_0dn_us.pkl', 'rb') as f:
        audio_emo_lb = pickle.load(f)

    audio_emo_classes = audio_emo_lb.classes_

    audio_emo_probability = get_audio_sentiment_emo(audio)

    # print('label audio_emo_classes: ', audio_emo_classes)
    # Get the final predicted label
    audio_emo_prob_index = audio_emo_probability.argmax(axis=1) # this outputs the highest index - example: [1]
    # print('audio_emo_probability.argmax: ', audio_emo_prob_index)

    # pred_label = audio_emo_classes[audio_emo_prob_index]
    # print('audio_emo_classes[audio_emo_prob_index]: ', pred_label)

    # final = final.astype(int).flatten()
    # print('prediction flatten: ', final)

    audio_emo_prediction = audio_emo_lb.inverse_transform((audio_emo_prob_index))
    # print('audio_emo_lb.inverse_transform of audio_emo_prob_index: ', prediction) 

    # print('trying to get out list of audio_emo_classes and its audio_emo_probability')
    # print('step 1 is to get probabilty list')
    audio_emo_prob_list = audio_emo_probability[0].tolist()
    # print('audio_emo_prob_list is: ', audio_emo_prob_list)

    audio_emo_class_prob = [(audio_emo_classes[i], audio_emo_prob_list[i]) for i in range(len(audio_emo_classes))]
    # print('audio_emo_classes and its audio_emo_probability: ', audio_emo_class_prob)

    # print({'label': audio_emo_prediction, 'audio_emo_probability': audio_emo_class_prob})

    results_dict['audio_emotion'] = audio_emo_prediction[0]
    results_dict['audio_emotion_prob'] = dict(audio_emo_class_prob)

    print('this is the final dictionary output')
    print(results_dict)



if __name__ == '__main__':
    # dotenv_path = join(dirname(__file__), '.env')
    # load_dotenv(dotenv_path)
    main()
    # try:
    #     main()
    # except:
    #     print("IOError detected, restarting...")
    #     main()
