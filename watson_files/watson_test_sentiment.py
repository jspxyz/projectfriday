import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions

authenticator = IAMAuthenticator('pC5Lg1_lf0msxrFZzgWzC5UZR7diA4sXEBWqs8JAI8kN')
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2020-08-01',
    authenticator=authenticator
)

natural_language_understanding.set_service_url('https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/13ff64aa-1811-4714-94bd-24780dcf9aa0')

response = natural_language_understanding.analyze(
    url='www.wsj.com/news/markets',
    features=Features(sentiment=SentimentOptions())).get_result()

print(json.dumps(response, indent=2))