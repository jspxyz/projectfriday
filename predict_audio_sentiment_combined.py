'''
This file is for Watson API and audio sentiment
Uses Recorder2 to save audio
Uses s2t and nlu api to get text, confidence of text, and sentiment.
Uses Audio MFCC40 to get audio sentiment.
Created on 20201214 0742
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
        speech_recognition_results = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav'
            # word_alternatives_threshold=0.9,
            # keywords=['colorado', 'tornado', 'tornadoes'],
            # keywords_threshold=0.5
        ).get_result()
    # print(type(speech_recognition_results))
    # found that speech_recognition_results is a dictionary
    # removing the below because json.dumps converts into a string
    # transcribe_audio_result = json.dumps(speech_recognition_results, indent=2)
    return speech_recognition_results

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

    response = natural_language_understanding.analyze(
        # url='www.ibm.com',
        # text=' '.join(contents),
        text=text,
        features=Features(keywords=KeywordsOptions(sentiment=True,emotion=True,limit=2))).get_result()

    # print(json.dumps(response, indent=2))
    return response

# function to get audio sentiment
# code pulled from predict_audio_sentiment_mfcc40
def get_audio_sentiment(path_to_audio_file):
    print('Loading audio sentiment model...')
    # loading model with just h5
    # Recreate the exact same model, including its weights and the optimizer
    model = tf.keras.models.load_model('./models_saved/model_d_conv1d_mfcc40_0dn_us_pol_b32.h5')

    # print('Loaded model. Spinning up the recorder...')
    # # recorder for X seconds
    # # returns the wav file path
    # audio = record(5)
    audio = path_to_audio_file

    # print('This is what you recorded!')
    # os.system("afplay " + audio)

    print('Making prediction...')
    # convert into librosa
    # data, sampling_rate = librosa.load(audio)

    # Transform the file so we can apply the predictions
    X, sample_rate = librosa.load(audio,
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
    probability = model.predict(mfccs)
                            #  batch_size=16, 
                            #  verbose=1)

    # print('probability looks like: ', probability)

    # output prediction
    # filename = '/content/labels'
    # infile = open(filename,'rb')
    # lb = pickle.load(infile)
    # infile.close()
    with open('./Data_Array_Storage/labels_mfcc40_pol_0dn_us.pkl', 'rb') as f:
        lb = pickle.load(f)

    # print('label pickle file is: ', lb)

    classes = lb.classes_

    # print('label classes: ', classes)
    # Get the final predicted label
    prob_index = probability.argmax(axis=1) # this outputs the highest index - example: [1]
    # print('probability.argmax: ', prob_index)

    pred_label = classes[prob_index]
    # print('classes[prob_index]: ', pred_label)

    # final = final.astype(int).flatten()
    # print('prediction flatten: ', final)

    prediction = lb.inverse_transform((prob_index))
    # print('lb.inverse_transform of prob_index: ', prediction) 

    # print('trying to get out list of classes and its probability')
    # print('step 1 is to get probabilty list')
    prob_list = probability[0].tolist()
    # print('prob_list is: ', prob_list)

    class_prob = [(classes[i], prob_list[i]) for i in range(len(classes))]
    # print('classes and its probability: ', class_prob)

    print({'label': prediction, 'probability': class_prob})

    return prediction
    
def main():
    # recorder = Recorder("speech.wav")
    # record function from test_recorder2.py
    # returns WAVE_OUTPUT_FILENAME
    print('Spinning up the recorder...')
    print('Record now, please!')
    audio = record(5)

    print('This is what you recorded!')
    os.system("afplay " + audio)

    print("Transcribing audio....\n")
    # result = transcribe_audio('speech.wav')
    transcribe_audio_result = transcribe_audio(audio)


    print('-'*10)
    print('This is the Text Section')
    text = transcribe_audio_result['results'][0]['alternatives'][0]['transcript']
    text_confidence = transcribe_audio_result['results'][0]['alternatives'][0]['confidence']
    print("Text: " + text + "\n")
    print("Confidence: " + str(text_confidence) + "\n")
    
    print('-'*10)
    print("Analyzing text sentiment...\n")
    text_sentiment_results = get_text_sentiment(text)
    print('The text sentiment results are: ', text_sentiment_results)

    text_sentiment_polarity = text_sentiment_results['keywords'][0]['sentiment']['label']
    print('text sentiment polarity is: ', text_sentiment_polarity)
    text_sentiment_emotion_dict = text_sentiment_results['keywords'][0]['emotion']
    text_sentiment_emotion = max(text_sentiment_emotion_dict, key=text_sentiment_emotion_dict.get)
    print('text sentiment emotion is: ', text_sentiment_emotion)

    print('Let\'s predict your audio sentiment: ')
    audio_sentiment_polarity = get_audio_sentiment(audio)

    print('In summary:\n')
    print('This is what you said: ', text)
    print('Your text sentiment polarity is: ', text_sentiment_polarity)
    print('Your text sentiment emotion is: ', text_sentiment_emotion)
    print('Your audio sentiment polarity is: ', audio_sentiment_polarity)

if __name__ == '__main__':
    # dotenv_path = join(dirname(__file__), '.env')
    # load_dotenv(dotenv_path)
    main()
    # try:
    #     main()
    # except:
    #     print("IOError detected, restarting...")
    #     main()


# debugging print code below
    # print('-'*10)
    # print(type(transcribe_audio_result))
    # print(len(transcribe_audio_result))
    # print('-'*10)

    # print(transcribe_audio_result)
    # print('-'*10)
    # print(transcribe_audio_result['results'])
    # print(transcribe_audio_result['results'][0])
    # print(transcribe_audio_result['results'][0]['alternatives'])
    # print(transcribe_audio_result['results'][0]['alternatives'][0])
    # print(transcribe_audio_result['results'][0]['alternatives'][0]['transcript'])
    # print('-'*10)