from keras.datasets import imdb

training_set, testing_set = imdb.load_data(index_from=3)
X_train, Y_train = training_set
X_test, Y_test = testing_set


word_to_id = imdb.get_word_index()
word_to_id = {key: (value+3) for key, value in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
id_to_word = {value: key for key, value in word_to_id.items()}

print(' '.join(id_to_word[id] for id in X_train[6]))
print(Y_train[6])  # 0 이면 부정 1이면 긍정
