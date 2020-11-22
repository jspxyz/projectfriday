# 
# libraries
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# calling functions
from functions import *

# assigning csv
ref = pd.read_csv('./Data_path.csv')

# WARNING: THIS TAKES FOREVER

df = pd.DataFrame(columns=['feature'])
df_noise = pd.DataFrame(columns=['feature'])
# df_speedpitch = pd.DataFrame(columns=['feature'])
cnt = 0

# loop feature extraction over the entire dataset
for i in tqdm(ref.path):
    
    # first load the audio 
    X, sample_rate = librosa.load(i,
                                  res_type='kaiser_fast',
                                  duration=2.5,
                                  sr=44100,
                                  offset=0.5)

    # take mfcc and mean as the feature. Could do min and max etc as well. 
    mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                        sr=np.array(sample_rate), 
                                        n_mfcc=13),
                    axis=0)
    
    df.loc[cnt] = [mfccs]   

    # random shifting (omit for now)
    # Stretch
    # pitch (omit for now)
    # dyn change
    
    # noise 
    aug = noise(X)
    aug = np.mean(librosa.feature.mfcc(y=aug, 
                                    sr=np.array(sample_rate), 
                                    n_mfcc=13),    
                  axis=0)
    df_noise.loc[cnt] = [aug]

    # speed pitch
    # aug = speedNpitch(X)
    # aug = np.mean(librosa.feature.mfcc(y=aug, 
    #                                 sr=np.array(sample_rate), 
    #                                 n_mfcc=13),    
    #               axis=0)
    # df_speedpitch.loc[cnt] = [aug]   

    cnt += 1