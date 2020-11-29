import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions

# changing system path to import config modeul
import os, sys
sys.path.insert(1, os.getcwd())
from config import config_nlu_api, config_nlu_api_url

api = config_nlu_api
api_url = config_nlu_api_url

# print(api)
# print(api_url)

authenticator = IAMAuthenticator(api)
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2020-08-01',
    authenticator=authenticator
)

natural_language_understanding.set_service_url(api_url)

path_to_file = './entries_txt/20201126_TTG1188_otter.ai.txt'

with open(path_to_file) as f:
    contents = f.readlines()

response = natural_language_understanding.analyze(
    # url='www.ibm.com',
    text=' '.join(contents),
    features=Features(keywords=KeywordsOptions(sentiment=True,emotion=True,limit=2))).get_result()

print(json.dumps(response, indent=2))