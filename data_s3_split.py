import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# calling the Data Path file
ref_data_path = pd.read_csv('./Data_path.csv')

# assigning the pickle files
with open('./Data_Array_Storage/data_features.pkl', 'rb') as f:
    df_features = pickle.load(f)

# saving df_features_noise as pickle file
with open('./Data_Array_Storage/data_features_noise.pkl', 'rb') as f:
    df_features_noise = pickle.load(f)

# combining dataframes
df_path_features = pd.concat([ref_data_path,pd.DataFrame(df_features)],axis=1)
df_path_features_noise = pd.concat([ref_data_path,pd.DataFrame(df_features_noise)],axis=1)

df_features_all = pd.concat([df_path_features,df_path_features_noise],axis=0,sort=False) # ,df_speedpitch
df_final = df_features_all.fillna(0)

# Split between train and test 
X_train, X_test, y_train, y_test = train_test_split(df_final.drop(['gender','emotion','label_emotion','polarity','label_polarity','path','source'],axis=1),
                                                    df_final.label_polarity,
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
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# Pickel the lb object for future use 
filename = 'labels'
outfile = open(filename,'wb')
pickle.dump(lb,outfile)
outfile.close()

with open('example.pkl', 'wb') as f:
    pickle.dump(df, f)

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)