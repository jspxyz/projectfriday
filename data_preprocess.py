# import libraries
import os
import pandas as pd

# Saving file paths
CREMA = './data/cremad/AudioWAV/'
RAV = './data/ravdess/audio_speech_actors_01-24/'
SAVEE = './data/savee/ALL/'
TESS = './data/tess/TESS Toronto emotional speech set data/'

# defining emotion list
emotion_neutral_list = ['calm', 'neutral', 'surprise']
emotion_positive_list = ['happy']
emotion_negative_list = ['angry', 'disgust', 'fear', 'sad']

# function to find polarity of emotions
def find_polarity(x):
  if x in emotion_neutral_list: 
    return 'neutral'
  elif x in emotion_positive_list: 
    return 'positive'
  elif x in emotion_negative_list:
    return 'negative'
  else:
    return 'neutral'

# CREMA-D
# create Crema Dataframe
dir_list_crema = os.listdir(CREMA)
dir_list_crema.sort()
# print(dir_list_crema[0:10])

# crema empty lists
gender_crema = []
emotion_crema = []
path_crema = []
female_crema = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,
          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]

for i in dir_list_crema: 
    part = i.split('_')
    if int(part[0]) in female_crema:
        temp = 'female'
    else:
        temp = 'male'
    # gender_crema.append(temp)
    if part[2] == 'SAD' and temp == 'male':
        emotion_crema.append('sad')
        gender_crema.append('male')
    elif part[2] == 'ANG' and temp == 'male':
        emotion_crema.append('angry')
        gender_crema.append('male')
    elif part[2] == 'DIS' and temp == 'male':
        emotion_crema.append('disgust')
        gender_crema.append('male')
    elif part[2] == 'FEA' and temp == 'male':
        emotion_crema.append('fear')
        gender_crema.append('male')
    elif part[2] == 'HAP' and temp == 'male':
        emotion_crema.append('happy')
        gender_crema.append('male')
    elif part[2] == 'NEU' and temp == 'male':
        emotion_crema.append('neutral')
        gender_crema.append('male')
    elif part[2] == 'SAD' and temp == 'female':
        emotion_crema.append('sad')
        gender_crema.append('female')
    elif part[2] == 'ANG' and temp == 'female':
        emotion_crema.append('angry')
        gender_crema.append('female')
    elif part[2] == 'DIS' and temp == 'female':
        emotion_crema.append('disgust')
        gender_crema.append('female')
    elif part[2] == 'FEA' and temp == 'female':
        emotion_crema.append('fear')
        gender_crema.append('female')
    elif part[2] == 'HAP' and temp == 'female':
        emotion_crema.append('happy')
        gender_crema.append('female')
    elif part[2] == 'NEU' and temp == 'female':
        emotion_crema.append('neutral')
        gender_crema.append('female')
    else:
        emotion_crema.append('unknown')
    path_crema.append(CREMA + i)

# creating gender, emotion dataframe
CREMA_df = pd.DataFrame(gender_crema, columns = ['gender'])
CREMA_df = pd.concat([CREMA_df, pd.DataFrame(emotion_crema, columns = ['emotion'])],axis=1)
CREMA_df['label_emotion'] = CREMA_df.gender + '_' + CREMA_df.emotion

# assign polarity
CREMA_df['polarity'] = pd.Series([0])
CREMA_df['polarity'] = CREMA_df['emotion'].apply(find_polarity)
CREMA_df['label_polarity'] = CREMA_df.gender + '_' + CREMA_df.polarity

# add source info
CREMA_df['source'] = 'CREMA'
CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path_crema, columns = ['path'])],axis=1)

# for testing
# print(CREMA_df.label_polarity.value_counts())
# misc. code

