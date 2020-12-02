import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
X_train, X_test, y_train, y_test = train_test_split(df_final.drop(['gender','emotion','label_emotion','polarity','label_polarity','path','source'],axis=1),
                                                    df_final.polarity,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    random_state=42)

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

# save y_train, y_test
with open('./Data_Array_Storage/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('./Data_Array_Storage/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

# Pickel the lb object for future use 
with open('./Data_Array_Storage/labels.pkl', 'wb') as f:
    pickle.dump(lb, f)

# expanding X_train and X_test dimensions
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# saving X_train and X_test
with open('./Data_Array_Storage/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('./Data_Array_Storage/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

# testing when could not convert numpy to tensor
# due to missing features tolist portion
# X_train_tensor = tf.convert_to_tensor(X_train)

# print(type(X_train_tensor))

# with open('./Data_Array_Storage/X_train.npy', 'wb') as f:
#     np.save(f, X_train)

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
    # pickle.dump(lb,outfile)
    # outfile.close()
