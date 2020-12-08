'''
This file is for Watson API.
Uses Recorder2 to save audio and uses s2t to get text.
Last Updated: 2020.12.01
Better working version in speech_sentimnet_run2.py
Can archive this file.
'''

import os
import json
from os.path import join, dirname
# from dotenv import load_dotenv
# from watson_developer_cloud import SpeechToTextV1 as SpeechToText
# from watson_developer_cloud import AlchemyLanguageV1 as AlchemyLanguage

from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# from speech_sentiment_python.recorder import Recorder
from recorder2 import record

# changing system path to import config modeul
import os, sys
sys.path.insert(1, os.getcwd())
from config import config_s2t_api, config_s2t_api_url

api = config_s2t_api
api_url = config_s2t_api_url

def transcribe_audio(path_to_audio_file):
    authenticator = IAMAuthenticator(api)
    speech_to_text = SpeechToTextV1(
        authenticator=authenticator
    )

    speech_to_text.set_service_url(api_url)

    # username = os.environ.get("BLUEMIX_USERNAME")
    # password = os.environ.get("BLUEMIX_PASSWORD")
    # speech_to_text = SpeechToText(username=username,
    #                               password=password)

    with open(join(dirname(__file__), path_to_audio_file), 'rb') as audio_file:
        return speech_to_text.recognize(audio_file,
            content_type='audio/wav')

# def get_text_sentiment(text):
#     alchemy_api_key = os.environ.get("ALCHEMY_API_KEY")
    
#     alchemy_language = AlchemyLanguage(api_key=alchemy_api_key)
#     result = alchemy_language.sentiment(text=text)
#     if result['docSentiment']['type'] == 'neutral':
#         return 'netural', 0
#     return result['docSentiment']['type'], result['docSentiment']['score']

def main():
    # recorder = Recorder("speech.wav")
    audio = record()

    # print("Please say something nice into the microphone\n")
    # recorder.record_to_file()

    print("Transcribing audio....\n")
    # result = transcribe_audio('speech.wav')
    result = transcribe_audio(audio)

    text = result['results'][0]['alternatives'][0]['transcript']
    print("Text: " + text + "\n")
    
    # sentiment, score = get_text_sentiment(text)
    # print(sentiment, score)  

if __name__ == '__main__':
    # dotenv_path = join(dirname(__file__), '.env')
    # load_dotenv(dotenv_path)
    try:
        main()
    except:
        print("IOError detected, restarting...")
        main()