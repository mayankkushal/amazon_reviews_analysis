import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
# Importing the dataset
dataset = pd.read_csv('Reviews.csv')

num_words = 10000

tokenizer = Tokenizer(num_words=num_words)
# Mapping every text to sequence of word embeddings
tokenizer.fit_on_texts(dataset['Text'])
X = tokenizer.texts_to_sequences(dataset['Text'])
y = dataset.iloc[:, 6].values

y[y <= 3] = 0
y[y > 3] = 1

X_train, X_test, y_train, y_test = train_test_split(X[:300000], y[:300000], test_size = 0.30)

max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Using embedding from Keras
embedding_vecor_length = 300
model = Sequential()
model.add(Embedding(num_words, embedding_vecor_length, input_length=max_review_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same', activation='relu'))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=3, batch_size=64)

# Evaluation on the test set
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))