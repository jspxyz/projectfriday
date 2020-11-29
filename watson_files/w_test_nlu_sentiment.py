import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

# changing system path to import config modeul
import os, sys
sys.path.insert(1, os.getcwd())
from config import config_nlu_api, config_nlu_api_url

api = config_nlu_api
api_url = config_nlu_api_url

authenticator = IAMAuthenticator(api)
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2020-08-01',
    authenticator=authenticator
)

natural_language_understanding.set_service_url(api_url)

response = natural_language_understanding.analyze(
    url='www.wsj.com/news/markets',
    features=Features(sentiment=SentimentOptions())).get_result()

print(json.dumps(response, indent=2))