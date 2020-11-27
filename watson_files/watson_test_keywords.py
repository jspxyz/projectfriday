import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions

api = 'pC5Lg1_lf0msxrFZzgWzC5UZR7diA4sXEBWqs8JAI8kN'
api_url = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/13ff64aa-1811-4714-94bd-24780dcf9aa0'

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