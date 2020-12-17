'''
Data Process Step 1
Creating a CSV file that has all files with following data:
emotion, gender, emotion_label, polarity, polarity_label, dataset source, filepath

Major Change Log:
20201209 1708 - changed positive polarity to include surprise
    - reasoning: surprising negative output would be defined as negative
'''

# import libraries
import os
import pandas as pd

# Saving file paths
CREMA = './data/cremad/AudioWAV/'
RAV = './data/ravdess/audio_speech_actors_01-24/'
SAVEE = './data/savee/ALL/'
TESS = './data/tess/TESS Toronto emotional speech set data/'

# defining emotion list
emotion_neutral_list = ['calm', 'neutral']
emotion_positive_list = ['happy', 'surprise']
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
## CREMA dataframe start ##
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
## CREMA dataframe end ##


# Create RAV dataframe
## RAV dataframe start ##
dir_list_rav = os.listdir(RAV)
dir_list_rav.sort()

# empty lists
emotion_rav = []
gender_rav = []
path_rav = []

for i in dir_list_rav:
    fname = os.listdir(RAV + i)
    for f in fname:
        part = f.split('.')[0].split('-')
        emotion_rav.append(int(part[2]))
        temp = int(part[6])
        if temp%2 == 0:
            temp = "female"
        else:
            temp = "male"
        gender_rav.append(temp)
        path_rav.append(RAV + i + '/' + f)

# creating emotion and gender dataframe
RAV_df = pd.DataFrame(emotion_rav)
RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
RAV_df = pd.concat([pd.DataFrame(gender_rav),RAV_df],axis=1)
RAV_df.columns = ['gender', 'emotion']
RAV_df['label_emotion'] = RAV_df.gender + '_' + RAV_df.emotion

# finding polarity
RAV_df['polarity'] = pd.Series([0])
RAV_df['polarity'] = RAV_df['emotion'].apply(find_polarity) # , axis=1
RAV_df['label_polarity'] = RAV_df.gender + '_' + RAV_df.polarity

# adding source
RAV_df['source'] = 'RAVDESS'  
RAV_df = pd.concat([RAV_df,pd.DataFrame(path_rav, columns = ['path'])],axis=1)

# misc. code
# RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)
# RAV_df.label_emotion.value_counts()

# code for testing
# print(RAV_df.head())

## RAV dataframe end ##

# Get the data location for SAVEE
dir_list_savee = os.listdir(SAVEE)

# parse the filename to get the emotions
emotion_savee = []
polarity_savee = []
path_savee = []
gender_savee = []
for i in dir_list_savee:
    if i[-8:-6]=='_a':
        emotion_savee.append('angry')
        gender_savee.append('male')
    elif i[-8:-6]=='_d':
        emotion_savee.append('disgust')
        gender_savee.append('male')
    elif i[-8:-6]=='_f':
        emotion_savee.append('fear')
        gender_savee.append('male')
    elif i[-8:-6]=='_h':
        emotion_savee.append('happy')
        gender_savee.append('male')
    elif i[-8:-6]=='_n':
        emotion_savee.append('neutral')
        gender_savee.append('male')
    elif i[-8:-6]=='sa':
        emotion_savee.append('sad')
        gender_savee.append('male')
    elif i[-8:-6]=='su':
        emotion_savee.append('surprise')
        gender_savee.append('male')
    else:
        emotion_savee.append('unknown')
        gender_savee.append('male')
    path_savee.append(SAVEE + i)
    
# creating gender, emotion dataframe
SAVEE_df = pd.DataFrame(gender_savee, columns = ['gender'])
SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(emotion_savee, columns = ['emotion'])],axis=1)
SAVEE_df['label_emotion'] = SAVEE_df.gender + '_' + SAVEE_df.emotion

# assign polarity
SAVEE_df['polarity'] = pd.Series([0])
SAVEE_df['polarity'] = SAVEE_df['emotion'].apply(find_polarity)
SAVEE_df['label_polarity'] = SAVEE_df.gender + '_' + SAVEE_df.polarity

# add source and path
SAVEE_df['source'] = 'SAVEE'
SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path_savee, columns = ['path'])], axis = 1)

# testing 
# print(SAVEE_df.label_polarity.value_counts())
# print(SAVEE_df.sample(10))

# Misc. code
# SAVEE_df['gender'] = 'male'
# SAVEE_df.labels.value_counts()


## TESS dataframe start
# creating TESS dataframe
dir_list_tess = os.listdir(TESS)
dir_list_tess.sort()
# dir_list_tess

path_tess = []
emotion_tess = []
gender_tess = []

for i in dir_list_tess:
    if i == '.DS_Store': # ignoring the stupid mac file
        continue

    fname = os.listdir(TESS + i)
    
    for f in fname:
        if i == 'OAF_angry' or i == 'YAF_angry':
            emotion_tess.append('angry')
            gender_tess.append('female')
        elif i == 'OAF_disgust' or i == 'YAF_disgust':
            emotion_tess.append('disgust')
            gender_tess.append('female')
        elif i == 'OAF_Fear' or i == 'YAF_fear':
            emotion_tess.append('fear')
            gender_tess.append('female')
        elif i == 'OAF_happy' or i == 'YAF_happy':
            emotion_tess.append('happy')
            gender_tess.append('female')
        elif i == 'OAF_neutral' or i == 'YAF_neutral':
            emotion_tess.append('neutral')
            gender_tess.append('female')                                
        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
            emotion_tess.append('surprise')
            gender_tess.append('female')
        elif i == 'OAF_Sad' or i == 'YAF_sad':
            emotion_tess.append('sad')
            gender_tess.append('female')
        else:
            emotion_tess.append('unknown')
            gender_tess.append('female')
        path_tess.append(TESS + i + "/" + f)


# # creating gender, emotion dataframe
# TESS_df = pd.DataFrame(gender_savee, columns = ['gender'])
# TESS_df = pd.concat([TESS_df, pd.DataFrame(emotion_savee, columns = ['emotion'])],axis=1)
# TESS_df['label_emotion'] = TESS_df.gender + '_' + TESS_df.emotion

# creating gender, emotion dataframe
TESS_df = pd.DataFrame(gender_tess, columns = ['gender'])
TESS_df = pd.concat([TESS_df, pd.DataFrame(emotion_tess, columns = ['emotion'])], axis=1)
TESS_df['label_emotion'] = TESS_df.gender + '_' + TESS_df.emotion

# assigning polarity
TESS_df['polarity'] = pd.Series([0])
TESS_df['polarity'] = TESS_df['emotion'].apply(find_polarity)
TESS_df['label_polarity'] = TESS_df.gender + '_' + TESS_df.polarity

# assigning source and path
TESS_df['source'] = 'TESS'
TESS_df = pd.concat([TESS_df,pd.DataFrame(path_tess, columns = ['path'])],axis=1)

# misc code
# TESS_df.head()
# TESS_df['gender'] = 'female'

# testing
# print(TESS_df.label_polarity.value_counts())
# print(TESS_df.sample(10))

## TESS Dataframe end ##

## combine dataframes
df_path = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis = 0)
# print(df.label_polarity.value_counts())
# df.head()
# print(df_path.shape)
# df_path.to_csv("./Data_Array_Storage/Data_path.csv",index=False)

df_male = df_path[df_path['gender'] == 'male']

print(df_male[:5])

print(df_male.shape)

print(df_male.polarity.value_counts())
print(df_male.emotion.value_counts())

df_male.to_csv("./Data_Array_Storage/Data_male_path.csv",index=False)