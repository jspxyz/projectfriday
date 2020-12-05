'''
- Data Processing Part 3
- Applying Oversampling to dataset to compensate for low Positive polarity
'''

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

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
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(df_final.drop(['gender','emotion','label_emotion','polarity','label_polarity','path','source'],axis=1),
                                                    df_final.polarity,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42)

print('Shape after split into train and test')
print('X_train: ', X_train_smote.shape)
print('X_test: ', X_test_smote.shape)
print('y_train: ', y_train_smote.shape)
print('y_test: ', y_test_smote.shape)

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
smote = SMOTE(sampling_strategy='minority',
                                random_state=42)

# fit and apply the transform
X_train_smote, y_train_smote = smote.fit_resample(X_train_smote, y_train_smote)

print('Shape after SMOTE')
print('X_train: ', X_train_smote.shape)
print('X_test: ', X_test_smote.shape)
print('y_train: ', y_train_smote.shape)
print('y_test: ', y_test_smote.shape)


# Data normalization 
mean = np.mean(X_train_smote, axis=0)
std = np.std(X_train_smote, axis=0)

X_train_smote = (X_train_smote - mean)/std
X_test_smote = (X_test_smote - mean)/std

# Preparation steps to get it into the correct format for Keras 
X_train_smote = np.array(X_train_smote)
y_train_smote = np.array(y_train_smote)
X_test_smote = np.array(X_test_smote)
y_test_smote = np.array(y_test_smote)

# one hot encode the target 
lb_smote = LabelEncoder()
y_train_smote = tf.keras.utils.to_categorical(lb_smote.fit_transform(y_train_smote)) # tf.keras.utils.to_categorical
y_test_smote = tf.keras.utils.to_categorical(lb_smote.fit_transform(y_test_smote))

# save y_train_smote, y_test_smote
with open('./Data_Array_Storage/y_train_smote.pkl', 'wb') as f:
    pickle.dump(y_train_smote, f)

with open('./Data_Array_Storage/y_test_smote.pkl', 'wb') as f:
    pickle.dump(y_test_smote, f)

# Pickel the lb_smote object for future use 
with open('./Data_Array_Storage/labels_smote.pkl', 'wb') as f:
    pickle.dump(lb_smote, f)

# expanding X_train_smote and X_test_smote dimensions
X_train_smote = np.expand_dims(X_train_smote, axis=2)
X_test_smote = np.expand_dims(X_test_smote, axis=2)

# saving X_train_smote and X_test_smote
with open('./Data_Array_Storage/X_train_smote.pkl', 'wb') as f:
    pickle.dump(X_train_smote, f)

with open('./Data_Array_Storage/X_test_smote.pkl', 'wb') as f:
    pickle.dump(X_test_smote, f)

print('Shape after format for keras')
print('X_train: ', X_train_smote.shape)
print('X_test: ', X_test_smote.shape)
print('y_train: ', y_train_smote.shape)
print('y_test: ', y_test_smote.shape)

