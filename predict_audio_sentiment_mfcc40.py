'''
This file uses MFCC40 to predict audio sentiment
Updated on 20201214 0753
Update includes moving sentiment to functions
'''

import librosa
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from recorder2 import record

# from flask import Flask, jsonify

# load the model
# code from predict_load_model
# from models import model_a_conv1d

# assigning the pickle files
# with open('./Data_Array_Storage/X_train.pkl', 'rb') as f:
#     X_train = pickle.load(f)

# print('Opening pickle files...')
# with open('./Data_Array_Storage/X_test_mfcc40_pol_0dn_us.pkl', 'rb') as f:
#     X_test = pickle.load(f)

# with open('./Data_Array_Storage/y_train.pkl', 'rb') as f:
#     y_train = pickle.load(f)

# with open('./Data_Array_Storage/y_test_mfcc40_pol_0dn_us.pkl', 'rb') as f:
#     y_test = pickle.load(f)

def get_audio_sentiment(path_to_audio_file):
    print('Loading audio sentiment model...')
    # loading model with just h5
    # Recreate the exact same model, including its weights and the optimizer
    model = tf.keras.models.load_model('./models_saved/model_d_conv1d_mfcc40_0dn_us_pol_b32.h5')

    # print('Loaded model. Spinning up the recorder...')
    # # recorder for X seconds
    # # returns the wav file path
    # audio = record(5)
    audio = path_to_audio_file

    # print('This is what you recorded!')
    # os.system("afplay " + audio)

    print('Making prediction...')
    # convert into librosa
    # data, sampling_rate = librosa.load(audio)

    # Transform the file so we can apply the predictions
    X, sample_rate = librosa.load(audio,
                                res_type='kaiser_best',
                                duration=2.5,
                                sr=44100,
                                offset=0.5
                                )


    mfccs = librosa.feature.mfcc(y=X, 
                                sr=sample_rate,
                                n_mfcc=40)

    mfccs = np.moveaxis(mfccs, 0, -1)

    print('mfccs looks like: ', mfccs)
    print('mfccs shape is: ', mfccs.shape)
    # newdf = pd.DataFrame(data=mfccs).T

    mfccs = np.expand_dims(mfccs, axis=0)
    print('mfccs expanded shape is: ', mfccs.shape)

    # apply prediction
    # newdf= np.expand_dims(newdf, axis=2)
    probability = model.predict(mfccs)
                            #  batch_size=16, 
                            #  verbose=1)

    print('probability looks like: ', probability)

    # output prediction
    # filename = '/content/labels'
    # infile = open(filename,'rb')
    # lb = pickle.load(infile)
    # infile.close()
    with open('./Data_Array_Storage/labels_mfcc40_pol_0dn_us.pkl', 'rb') as f:
        lb = pickle.load(f)

    print('label pickle file is: ', lb)

    classes = lb.classes_

    print('label classes: ', classes)
    # Get the final predicted label
    prob_index = probability.argmax(axis=1) # this outputs the highest index - example: [1]
    print('probability.argmax: ', prob_index)

    pred_label = classes[prob_index]
    print('classes[prob_index]: ', pred_label)

    # final = final.astype(int).flatten()
    # print('prediction flatten: ', final)

    prediction = lb.inverse_transform((prob_index))
    print('lb.inverse_transform of prob_index: ', prediction) 

    print('trying to get out list of classes and its probability')
    print('step 1 is to get probabilty list')
    prob_list = probability[0].tolist()
    print('prob_list is: ', prob_list)

    class_prob = [(classes[i], prob_list[i]) for i in range(len(classes))]
    print('classes and its probability: ', class_prob)

    print({'label': prediction, 'probability': class_prob})

    return prediction

def main():
    print('Spinning up the recorder...')
    # recorder for X seconds
    # returns the wav file path
    audio = record(5)

    print('This is what you recorded!')
    os.system("afplay " + audio)

    print('Let\'s predict your audio sentiment: ')
    get_audio_sentiment(audio)




if __name__ == '__main__':
    # dotenv_path = join(dirname(__file__), '.env')
    # load_dotenv(dotenv_path)
    main()
    # try:
    #     main()
    # except:
    #     print("IOError detected, restarting...")
    #     main()