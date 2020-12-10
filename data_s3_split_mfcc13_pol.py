'''
Data Processing Part 3
Splitting with MFCC 13 features
Labels: Polarity
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
with open('./Data_Array_Storage/data_features_mfcc13.pkl', 'rb') as f:
    df_features = pickle.load(f)

# opening df_features_noise as pickle file
with open('./Data_Array_Storage/data_features_noise_mfcc13.pkl', 'rb') as f:
    df_features_noise = pickle.load(f)

# opening error_list as pickle file
with open('./Data_Array_Storage/error_list_mfcc13.pkl', 'rb') as f:
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
# keep polarity
y_full = df_y_full.drop(['gender','emotion','label_emotion','label_polarity','path','source'],axis=1).to_numpy().squeeze()

print('y_full after drop: ', y_full[:5])

# set random seed
np.random.seed(42)

# set indices to randomize
indices = np.random.permutation(len(X_full))
train_size = 0.8
len_train_set = int(len(X_full) * train_size)

X_shuffle = X_full[indices]
y_shuffle = y_full[indices]
X_train = X_shuffle[:len_train_set]
y_train = y_shuffle[:len_train_set]

X_test = X_shuffle[len_train_set:]
y_test = y_shuffle[len_train_set:]

print('shapes after np.random.permutation splits')
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test: ', X_test.shape)
print('y_test: ', y_test.shape)

print('y_test header 5')
print(y_test[:5])

print('X_test header 3')
print(X_test[:3])

# combining path with features
# changing features to list
# df_path_features = pd.concat([ref_data_path,pd.DataFrame(df_features['feature'].values.tolist())],axis=1)
# df_path_features_noise = pd.concat([ref_data_path,pd.DataFrame(df_features_noise['feature'].values.tolist())],axis=1)

# df_features_all = pd.concat([df_path_features,df_path_features_noise],axis=0,sort=False) # ,df_speedpitch
# df_final = df_features_all.fillna(0)

# df_final = pd.concat([ref_data_path, pd.DataFrame(df_features)],axis=1)

# # Split between train and test 
# X_train, X_test, y_train, y_test = train_test_split(df_final.drop(['gender','emotion','label_emotion','polarity','label_polarity','path','source'],axis=1),
#                                                     df_final.polarity,
#                                                     test_size=0.25,
#                                                     shuffle=True,
#                                                     random_state=42)

# print('Shape after train_test_split')
# print('X_train: ', X_train.shape)
# print('X_test: ', X_test.shape)
# print('y_train: ', y_train.shape)
# print('y_test: ', y_test.shape)

# Data normalization 
# original 
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# new methood to data normazilie over each individual
# mean = np.mean(np.reshape(X_train, (X_train.shape[0], -1)), axis=1) # (1000,)
# std = np.std(np.reshape(X_train, (X_train.shape[0], -1)), axis=1)   # (1000,)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

# Preparation steps to get it into the correct format for Keras 
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# X_test = np.array(X_test)
# y_test = np.array(y_test)

print('Shape after data normalization for X_ only')
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)

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
with open('./Data_Array_Storage/y_train_mfcc13_axis0_pol.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('./Data_Array_Storage/y_test_mfcc13_axis0_pol.pkl', 'wb') as f:
    pickle.dump(y_test, f)

# Pickel the lb object for future use 
with open('./Data_Array_Storage/labels_mfcc13_axis0_pol.pkl', 'wb') as f:
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
with open('./Data_Array_Storage/X_train_mfcc13_axis0_pol.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('./Data_Array_Storage/X_test_mfcc13_axis0_pol.pkl', 'wb') as f:
    pickle.dump(X_test, f)

print('Pickle files saved. Final shpaes:')
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)