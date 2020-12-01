import json
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# changing system path to import config modeul
import os, sys
sys.path.insert(1, os.getcwd())
from config import config_s2t_api, config_s2t_api_url

api = config_s2t_api
api_url = config_s2t_api_url

authenticator = IAMAuthenticator(api)
speech_to_text = SpeechToTextV1(
    authenticator=authenticator
)

speech_to_text.set_service_url(api_url)

class MyRecognizeCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_data(self, data):
        print(json.dumps(data, indent=2))

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

myRecognizeCallback = MyRecognizeCallback()

# with open(join(dirname(__file__), './.', 'test_r2_testing.wav'), # audio-file.flac
#               'rb') as audio_file:


with open(join('./.', 'test_r2_testing.wav'),
              'rb') as audio_file:
    audio_source = AudioSource(audio_file)
    speech_to_text.recognize_using_websocket(
        audio=audio_source,
        content_type='audio/wav', # content_type='audio/flac',
        recognize_callback=myRecognizeCallback,
        model='en-US_BroadbandModel')
        # keywords=['colorado', 'tornado', 'tornadoes'],
        # keywords_threshold=0.5,
        # max_alternatives=3)