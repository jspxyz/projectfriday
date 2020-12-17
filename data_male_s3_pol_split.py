'''
Data Processing Part 3
Splitting with MFCC 40 features
undersampling, Data Normalization
Labels: Polarity
'''

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from collections import Counter
from imblearn.under_sampling import NearMiss

# calling the Data Path file
ref_data_path = pd.read_csv('./Data_Array_Storage/male_Data_path.csv')

# opening df_features
with open('./Data_Array_Storage/male_data_features.pkl', 'rb') as f:
    df_features = pickle.load(f)

# opening df_features_noise as pickle file
with open('./Data_Array_Storage/male_data_features_noise.pkl', 'rb') as f:
    df_features_noise = pickle.load(f)

# opening error_list as pickle file
with open('./Data_Array_Storage/male_error_list.pkl', 'rb') as f:
    error_list = pickle.load(f)

print("Error list: ", len(error_list))

# changing lists into numpy arrays
# ref_data_path_array = np.array(ref_data_path) # do not need this
df_features_array = np.array(df_features)
df_features_noise_array = np.array(df_features_noise)

# creating a y table that matches X table by doubling the ref_data_path
df_y_full = pd.concat((ref_data_path, ref_data_path),axis=0)
print('df_y_full: ', df_y_full.shape)

# creating X table of all dataset
X_full = np.concatenate((df_features, df_features_noise),axis=0)
print('X_full: ', X_full.shape)

# drop the columns
# 'gender','emotion','label_emotion', 'polarity', 'label_polarity','path','source'
# keep polarity by deleting from list below
y_full = df_y_full.drop(['gender', 'emotion','label_emotion', 'label_polarity','path','source'],axis=1).to_numpy().squeeze()

print('y_full after drop: ', y_full[:5])
print('y_full shape after drop: ', y_full.shape)

# reshape to undersample
X_full_2d = X_full.reshape((X_full.shape[0],-1))
print('X_full_2d shape: ', X_full_2d.shape)

# create undersampler
undersample = NearMiss (sampling_strategy = "not minority")

print('Starting undersampling...')
X_under, y_under = undersample.fit_resample(X_full_2d, y_full)

y_counter = Counter(y_under)
print('Information after undersampling')
print('y_under count: ', y_counter)

print('Under set shapes:')
print('X_under shape: ', X_under.shape)
print('y_under shape: ', y_under.shape)

X_train_2d, X_test_2d, y_train, y_test = train_test_split(X_under,
                                                     y_under,
                                                     test_size=0.1,
                                                     shuffle=True,
                                                     random_state=42)

print('Shapes after train_test_splits')
print('X_train_2d: ', X_train_2d.shape)
print('y_train: ', y_train.shape)
print('X_test_2d: ', X_test_2d.shape)
print('y_test: ', y_test.shape)

# need to reshape
print('Reshaping X back to original shape...')

X_train = X_train_2d.reshape((X_train_2d.shape[0], X_full.shape[1], X_full.shape[2]))
X_test = X_test_2d.reshape((X_test_2d.shape[0], X_full.shape[1], X_full.shape[2]))

print('Reshaped X sets are now: ')
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

# Data normalization 
# original 
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

with open('./Data_Array_Storage/male_pol_us_mean.pkl', 'wb') as f:
    pickle.dump(mean, f)

with open('./Data_Array_Storage/male_pol_us_std.pkl', 'wb') as f:
    pickle.dump(std, f)

# # new methood to data normazilie over each individual
# # mean = np.mean(np.reshape(X_train, (X_train.shape[0], -1)), axis=1) # (1000,)
# # std = np.std(np.reshape(X_train, (X_train.shape[0], -1)), axis=1)   # (1000,)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

# print('Shape after data normalization for X_ only')
# print('X_train: ', X_train.shape)
# print('X_test: ', X_test.shape)
# print('y_train: ', y_train.shape)
# print('y_test: ', y_test.shape)

# one hot encode the target 
print('one hot label y sets...')
lb = LabelEncoder()
y_train = tf.keras.utils.to_categorical(lb.fit_transform(y_train))
y_test = tf.keras.utils.to_categorical(lb.transform(y_test))

print('Shape after one hot encode (for y_ only)')
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

print('y_test top 5', y_test[:5])

# save y_train, y_test
with open('./Data_Array_Storage/male_pol_us_y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('./Data_Array_Storage/male_pol_us_y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

# Pickel the lb object for future use 
with open('./Data_Array_Storage/male_pol_us_labels.pkl', 'wb') as f:
    pickle.dump(lb, f)

# expanding X_train and X_test dimensions
# no need to do this for conv1d
# X_train = np.expand_dims(X_train, axis=-1)
# X_test = np.expand_dims(X_test, axis=-1)
# print('Shape after X dimension expansion')
# print('X_train: ', X_train.shape)
# print('X_test: ', X_test.shape)
# print('y_train: ', y_train.shape)
# print('y_test: ', y_test.shape)

# saving X_train and X_test
with open('./Data_Array_Storage/male_pol_us_X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('./Data_Array_Storage/male_pol_us_X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

print('Pickle files saved. Final shpaes:')
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)