'''
Data Processing Part 3
Splitting with MFCC 13 AVG features
Label: Emotion
'''

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# calling the Data Path file
ref_data_path = pd.read_csv('./Data_Array_Storage/Data_path.csv')

# opening df_features
with open('./Data_Array_Storage/data_features_mfccavg.pkl', 'rb') as f:
    df_features = pickle.load(f)

# opening df_features_noise as pickle file
with open('./Data_Array_Storage/data_features_noise_mfccavg.pkl', 'rb') as f:
    df_features_noise = pickle.load(f)

# combining path with features
# changing features to list
df_path_features = pd.concat([ref_data_path,pd.DataFrame(df_features['feature'].values.tolist())],axis=1)
df_path_features_noise = pd.concat([ref_data_path,pd.DataFrame(df_features_noise['feature'].values.tolist())],axis=1)

df_features_all = pd.concat([df_path_features,df_path_features_noise],axis=0,sort=False) # ,df_speedpitch
df_final = df_features_all.fillna(0)

# Split between train and test 
# assigning emotion as label
X_train, X_test, y_train, y_test = train_test_split(df_final.drop(['gender','emotion','label_emotion','polarity','label_polarity','path','source'],axis=1),
                                                    df_final.emotion,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42)

print(y_train[:5])
# Data normalization 
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

# Preparation steps to get it into the correct format for Keras 
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# one hot encode the target 
lb = LabelEncoder()
y_train = tf.keras.utils.to_categorical(lb.fit_transform(y_train)) # tf.keras.utils.to_categorical
y_test = tf.keras.utils.to_categorical(lb.fit_transform(y_test))

print('Shape after one hot encode (for y_ only)')
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

print('y_test top 5', y_test[:5])


# save y_train, y_test
with open('./Data_Array_Storage/y_train_mfccavg_axis0_emo.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('./Data_Array_Storage/y_test_mfccavg_axis0_emo.pkl', 'wb') as f:
    pickle.dump(y_test, f)

# Pickel the lb object for future use 
with open('./Data_Array_Storage/labels_mfccavg_axis0_emo.pkl', 'wb') as f:
    pickle.dump(lb, f)

# expanding X_train and X_test dimensions
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# saving X_train and X_test
with open('./Data_Array_Storage/X_train_mfccavg_axis0_emo.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('./Data_Array_Storage/X_test_mfccavg_axis0_emo.pkl', 'wb') as f:
    pickle.dump(X_test, f)

print('Shape after format for keras')
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

print(X_train[:10])
print(y_train[:10])