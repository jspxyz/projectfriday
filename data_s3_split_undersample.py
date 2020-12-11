'''
Data Processing Part 3
Applying Undersampling to dataset to compensate for low Positive polarity
Updated on 20201209 1608 to use mfcc40
'''

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from imblearn.under_sampling import RandomUnderSampler

# calling the Data Path file
ref_data_path = pd.read_csv('./Data_Array_Storage/Data_path.csv')

# opening df_features
with open('./Data_Array_Storage/data_features_mfcc40.pkl', 'rb') as f:
    df_features = pickle.load(f)

# opening df_features_noise as pickle file
with open('./Data_Array_Storage/data_features_noise_mfcc40.pkl', 'rb') as f:
    df_features_noise = pickle.load(f)

# opening error_list as pickle file
with open('./Data_Array_Storage/error_list_40.pkl', 'rb') as f:
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
y_full = df_y_full.drop(['gender','emotion','label_emotion','label_polarity','path','source'],axis=1).to_numpy().squeeze()

print('y_full after drop: ', y_full[:5])

# set random seed
np.random.seed(42)

# set indices to randomize
indices = np.random.permutation(len(X_full))
train_size = 0.8
len_train_set = int(len(X_full) * train_size)

X_train_full = X_full[:len_train_set]
y_train_full = y_full[:len_train_set]

X_test_under = X_full[len_train_set:]
y_test_under = y_full[len_train_set:]

print('shapes after np.random.permutation splits')
print('X_train: ', X_train_full.shape)
print('y_train: ', y_train_full.shape)
print('X_test: ', X_test_under.shape)
print('y_test: ', y_test_under.shape)

print('y_test header 5')
print(y_test_under[:5])

print('X_test header 3')
print(X_test_under[:3])


'''
Undersample methods below.
Two methods
1.) manual
2.) using imlearn library
manual undershuffle code below
'''
# Method 1: #
#############
# Shuffle the Dataset.
# shuffled_df = credit_df.sample(frac=1,random_state=4)

# # Put all the fraud class in a separate dataset.
# fraud_df = shuffled_df.loc[shuffled_df['Class'] == 1]

# #Randomly select 492 observations from the non-fraud (majority class)
# non_fraud_df = shuffled_df.loc[shuffled_df['Class'] == 0].sample(n=492,random_state=42)

# # Concatenate both dataframes again
# normalized_df = pd.concat([fraud_df, non_fraud_df])
############

# Method 2: #
# using imlearn undersampler
# does not work with 3D data
#############
# Trial 1: Using sampling_strategy = majority, accuracy 46%
# Trail 2: changing sampling_strategy = 0.5, this does not work for multi-class


# define undersample strategy
# undersample = RandomUnderSampler(sampling_strategy='majority',
#                                 random_state=42)

# # fit and apply the transform
# X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

print('Shape after undersample')
print('X_train: ', X_train_under.shape)
print('X_test: ', X_test_under.shape)
print('y_train: ', y_train_under.shape)
print('y_test: ', y_test_under.shape)


# Data normalization 
mean = np.mean(X_train_under, axis=0)
std = np.std(X_train_under, axis=0)

X_train_under = (X_train_under - mean)/std
X_test_under = (X_test_under - mean)/std

# Preparation steps to get it into the correct format for Keras 
X_train_under = np.array(X_train_under)
y_train_under = np.array(y_train_under)
X_test_under = np.array(X_test_under)
y_test_under = np.array(y_test_under)

# one hot encode the target 
lb_under = LabelEncoder()
y_train_under = tf.keras.utils.to_categorical(lb_under.fit_transform(y_train_under)) # tf.keras.utils.to_categorical
y_test_under = tf.keras.utils.to_categorical(lb_under.transform(y_test_under))

# save y_train_under, y_test_under
with open('./Data_Array_Storage/y_train_under.pkl', 'wb') as f:
    pickle.dump(y_train_under, f)

with open('./Data_Array_Storage/y_test_under.pkl', 'wb') as f:
    pickle.dump(y_test_under, f)

# Pickel the lb_under object for future use 
with open('./Data_Array_Storage/labels_under.pkl', 'wb') as f:
    pickle.dump(lb_under, f)

# saving X_train_under and X_test_under
with open('./Data_Array_Storage/X_train_under.pkl', 'wb') as f:
    pickle.dump(X_train_under, f)

with open('./Data_Array_Storage/X_test_under.pkl', 'wb') as f:
    pickle.dump(X_test_under, f)

print('Shape after format for keras')
print('X_train: ', X_train_under.shape)
print('X_test: ', X_test_under.shape)
print('y_train: ', y_train_under.shape)
print('y_test: ', y_test_under.shape)

