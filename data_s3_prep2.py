import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# assigning csv
ref = pd.read_csv('./Data_path.csv')

# combining dataframes
df = pd.concat([ref,pd.DataFrame(df['feature'].values.tolist())],axis=1)
df_noise = pd.concat([ref,pd.DataFrame(df_noise['feature'].values.tolist())],axis=1)

df = pd.concat([df,df_noise],axis=0,sort=False) # ,df_speedpitch
df = df.fillna(0)

# Split between train and test 
X_train, X_test, y_train, y_test = train_test_split(df.drop(['gender','emotion','label_emotion','polarity','label_polarity','path','source'],axis=1),
                                                    df.label_polarity,
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

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)