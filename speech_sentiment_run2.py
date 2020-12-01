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
from test_recorder2 import record

# changing system path to import config modeul
import os, sys
sys.path.insert(1, os.getcwd())
from config import config_s2t_api, config_s2t_api_url, config_nlu_api, config_nlu_api_url

s2t_api = config_s2t_api
s2t_api_url = config_s2t_api_url
nlu_api = config_nlu_api
nlu_api_url = config_nlu_api_url

# original function from speech_sentiment_run.py
# def transcribe_audio(path_to_audio_file):
#     authenticator = IAMAuthenticator(api)
#     speech_to_text = SpeechToTextV1(
#         authenticator=authenticator
#     )

#     speech_to_text.set_service_url(api_url)

#     # username = os.environ.get("BLUEMIX_USERNAME")
#     # password = os.environ.get("BLUEMIX_PASSWORD")
#     # speech_to_text = SpeechToText(username=username,
#     #                               password=password)

#     with open(join(dirname(__file__), path_to_audio_file), 'rb') as audio_file:
#         return speech_to_text.recognize(audio_file,
#             content_type='audio/wav')

# working function taken from w_test_s2t_request.py file
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

# original text sentiment analysis code from speech_sentiment_run
# def get_text_sentiment(text):
#     alchemy_api_key = os.environ.get("ALCHEMY_API_KEY")
    
#     alchemy_language = AlchemyLanguage(api_key=alchemy_api_key)
#     result = alchemy_language.sentiment(text=text)
#     if result['docSentiment']['type'] == 'neutral':
#         return 'netural', 0
#     return result['docSentiment']['type'], result['docSentiment']['score']

# code taken from w_test_nlu_keywords.py
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

    return response
    # print(json.dumps(response, indent=2))

def main():
    # recorder = Recorder("speech.wav")
    # record function from test_recorder2.py
    # returns WAVE_OUTPUT_FILENAME
    audio = record()

    # print("Please say something nice into the microphone\n")
    # recorder.record_to_file()

    print("Transcribing audio....\n")
    # result = transcribe_audio('speech.wav')
    transcribe_audio_result = transcribe_audio(audio)

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

    text = transcribe_audio_result['results'][0]['alternatives'][0]['transcript']
    confidence = transcribe_audio_result['results'][0]['alternatives'][0]['confidence']
    print("Text: " + text + "\n")
    print("Confidence: " + str(confidence) + "\n")
    
    print("Analyzing sentiment...\n")
    sentiment = get_text_sentiment(text)
    # sentiment, score = get_text_sentiment(text)
    print(sentiment)
    # print(sentiment, score)  

if __name__ == '__main__':
    # dotenv_path = join(dirname(__file__), '.env')
    # load_dotenv(dotenv_path)
    main()
    # try:
    #     main()
    # except:
    #     print("IOError detected, restarting...")
    #     main()