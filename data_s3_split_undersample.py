'''
- Data Processing Part 3
- Applying Undersampling to dataset to compensate for low Positive polarity
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
with open('./Data_Array_Storage/data_features.pkl', 'rb') as f:
    df_features = pickle.load(f)

# opening df_features_noise as pickle file
with open('./Data_Array_Storage/data_features_noise.pkl', 'rb') as f:
    df_features_noise = pickle.load(f)

# combining path with features
# changing features to list
df_path_features = pd.concat([ref_data_path,pd.DataFrame(df_features['feature'].values.tolist())],axis=1)
df_path_features_noise = pd.concat([ref_data_path,pd.DataFrame(df_features_noise['feature'].values.tolist())],axis=1)

df_features_all = pd.concat([df_path_features,df_path_features_noise],axis=0,sort=False) # ,df_speedpitch
df_final = df_features_all.fillna(0)

# Split between train and test 
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(df_final.drop(['gender','emotion','label_emotion','polarity','label_polarity','path','source'],axis=1),
                                                    df_final.polarity,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42)

print('Shape after split into train and test')
print('X_train: ', X_train_under.shape)
print('X_test: ', X_test_under.shape)
print('y_train: ', y_train_under.shape)
print('y_test: ', y_test_under.shape)

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
#############
# Trial 1: Using sampling_strategy = majority, accuracy 46%
# Trail 2: changing sampling_strategy = 0.5, this does not work for multi-class


# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority',
                                random_state=42)

# fit and apply the transform
X_train_under, y_train_under = undersample.fit_resample(X_train_under, y_train_under)

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
y_test_under = tf.keras.utils.to_categorical(lb_under.fit_transform(y_test_under))

# save y_train_under, y_test_under
with open('./Data_Array_Storage/y_train_under.pkl', 'wb') as f:
    pickle.dump(y_train_under, f)

with open('./Data_Array_Storage/y_test_under.pkl', 'wb') as f:
    pickle.dump(y_test_under, f)

# Pickel the lb_under object for future use 
with open('./Data_Array_Storage/labels_under.pkl', 'wb') as f:
    pickle.dump(lb_under, f)

# expanding X_train_under and X_test_under dimensions
X_train_under = np.expand_dims(X_train_under, axis=2)
X_test_under = np.expand_dims(X_test_under, axis=2)

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

# testing when could not convert numpy to tensor
# due to missing features tolist portion
# X_train_tensor = tf.convert_to_tensor(X_train_under)

# print(type(X_train_tensor))

# with open('./Data_Array_Storage/X_train_under.npy', 'wb') as f:
#     np.save(f, X_train_under)

# with open('test.npy', 'wb') as f:
#     np.save(f, np.array([1, 2]))
#     np.save(f, np.array([1, 3]))
# with open('test.npy', 'rb') as f:
#     a = np.load(f)
#     b = np.load(f)
# print(a, b)
# [1 2] [1 3]

# with open('example.pkl', 'wb') as f:
#     pickle.dump(df, f)

# example: saving df_features as pickle file
# with open('./Data_Array_Storage/data_features.pkl', 'wb') as f:
#     pickle.dump(df_features, f)

# old method to pickle file
    # filename = 'labels'
    # outfile = open(filename,'wb')
    # pickle.dump(lb_under,outfile)
    # outfile.close()