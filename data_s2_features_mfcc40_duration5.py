'''
Data Processing Step 2
Pulls MFCC 40 features from each wav file
Major update on 20201215 1823
duration is now 5 seconds
input shape will move from (216, 40) to (431, 40)
'''


# libraries
import librosa
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# calling functions
from functions import noise

# assigning csv
ref_data_path = pd.read_csv('./Data_Array_Storage/Data_path.csv')

# WARNING: THIS TAKES FOREVER

# df_features = pd.DataFrame(columns=['feature'])

# df_features_noise = pd.DataFrame(columns=['feature'])

df_features = []
df_features_noise = []

# df_features_speedpitch = pd.DataFrame(columns=['feature'])
# cnt = 0

error_list = []

duration = 5
sr = 44100
offset = 0.5
n_mfcc = 40
max_len = round(duration * sr / 512)

# loop feature extraction over the entire dataset
for i, path in enumerate(tqdm(ref_data_path.path)):
    try:
    # first load the audio 
        X, sample_rate = librosa.load(path,
                                    res_type='kaiser_best',
                                    duration=duration,
                                    sr=sr,
                                    offset=offset)

        # take all mfcc as feature
        mfccs = librosa.feature.mfcc(y=X, 
                                    sr= sample_rate, # np.array(sample_rate), 
                                    n_mfcc=n_mfcc)
                                    # axis=0) # removed np.mean
        # If maximum length exceeds mfcc lengths then pad the remaining ones
        # max_len = duration*sr/512
        if (max_len > mfccs.shape[1]):
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Else cutoff the remaining parts
        else:
            mfccs = mfccs[:, :max_len]
        
        # changing axis to be (time, features)
        mfccs = np.moveaxis(mfccs, 0, -1)
        # df_features.loc[i] = [mfccs]   
        df_features.append(mfccs)
        
        # noise 
        aug = noise(X)
        aug = librosa.feature.mfcc(y=aug, 
                                   sr= sample_rate, # np.array(sample_rate), 
                                   n_mfcc=n_mfcc)
        
        # If maximum length exceeds mfcc lengths then pad the remaining ones
        if (max_len > aug.shape[1]):
            pad_width = max_len - aug.shape[1]
            aug = np.pad(aug, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Else cutoff the remaining parts
        else:
            aug = aug[:, :max_len]
        
        # changing axis to be (time, features)
        aug = np.moveaxis(aug, 0, -1)
        # df_features_noise.loc[i] = [aug]
        df_features_noise.append(aug)

                # random shifting (omit for now)
        # Stretch
        # pitch (omit for now)
        # dyn change

        # speed pitch
        # aug = speedNpitch(X)
        # aug = np.mean(librosa.feature.mfcc(y=aug, 
        #                                 sr=np.array(sample_rate), 
        #                                 n_mfcc=13),    
        #               axis=0)
        # df_features_speedpitch.loc[cnt] = [aug]   

        # cnt += 1
    except Exception as err:
        error_list.append(path)
        print('Error in processing', err)

print('df_features shape: ', np.shape(df_features))
# saving df_features as pickle file
with open('./Data_Array_Storage/data_features_mfcc40_duration5.pkl', 'wb') as f:
    pickle.dump(df_features, f)

print('df_features_noise shape: ', np.shape(df_features_noise))
# saving df_features_noise as pickle file
with open('./Data_Array_Storage/data_features_noise_mfcc40_duration5.pkl', 'wb') as f:
    pickle.dump(df_features_noise, f)

print('error_list shape: ', np.shape(error_list))
# saving df_features_noise as pickle file
with open('./Data_Array_Storage/error_list_mfcc40_duration5.pkl', 'wb') as f:
    pickle.dump(error_list, f)

# misc. Code
# df_features.to_csv("./Data_CSV/Data_features.csv",index=False)

# Pickel the lb object for future use 
# filename = 'labels'
# outfile = open(filename,'wb')
# pickle.dump(lb,outfile)
# outfile.close()

# this if function makes this file to run ONLY when its called
# only need this if i'm importing from another py file
# if __name__ == "__main__":