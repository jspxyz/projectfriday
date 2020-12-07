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