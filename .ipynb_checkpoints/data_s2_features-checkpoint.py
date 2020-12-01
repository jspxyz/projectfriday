# 
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

df_features = pd.DataFrame(columns=['feature'])

# this if function makes this file to run ONLY when its called
# only need this if i'm importing from another py file
# if __name__ == "__main__":

df_features_noise = pd.DataFrame(columns=['feature'])
# df_features_speedpitch = pd.DataFrame(columns=['feature'])
# cnt = 0

# loop feature extraction over the entire dataset
for i, path in enumerate(tqdm(ref_data_path.path)):
    try:
    # first load the audio 
        X, sample_rate = librosa.load(path,
                                    res_type='kaiser_fast',
                                    duration=2.5,
                                    sr=44100,
                                    offset=0.5)

        # take mfcc and mean as the feature. Could do min and max etc as well. 
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=np.array(sample_rate), 
                                            n_mfcc=13),
                                            axis=0)
        
        df_features.loc[i] = [mfccs]   
        
        # noise 
        aug = noise(X)
        aug = np.mean(librosa.feature.mfcc(y=aug, 
                                        sr=np.array(sample_rate), 
                                        n_mfcc=13),    
                                        axis=0)
        df_features_noise.loc[i] = [aug]

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
        print('Error in processing', err)

# saving df_features as pickle file
with open('./Data_Array_Storage/data_features.pkl', 'wb') as f:
    pickle.dump(df_features, f)

# saving df_features_noise as pickle file
with open('./Data_Array_Storage/data_features_noise.pkl', 'wb') as f:
    pickle.dump(df_features_noise, f)

# misc. Code
# df_features.to_csv("./Data_CSV/Data_features.csv",index=False)

# Pickel the lb object for future use 
# filename = 'labels'
# outfile = open(filename,'wb')
# pickle.dump(lb,outfile)
# outfile.close()