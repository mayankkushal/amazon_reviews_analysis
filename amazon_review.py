# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Reviews.csv')

'''
# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 568454):
    print('\r', str(i)+"/568454", end="")
    review = re.sub('[^a-zA-Z]', ' ', dataset['Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    # write the processed corpus to a file
    with open('outfile', 'wb') as fp:
    pickle.dump(itemlist, fp)
'''

import pickle
with open ('corpus_processed', 'rb') as fp:
    corpus = pickle.load(fp)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 6].values

'''
for i in range(0, 132503):
    if y[i] <= 3:
        y[i] = 0
    else:
        y[i] = 1
'''
# effecient way of doing the above steps
y[y <= 3] = 0
y[y > 3] = 1

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[:300000], y[:300000], test_size = 0.30)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'gini', 
                                    n_jobs = -1, max_depth=25)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
model2 = MultinomialNB(alpha=0.1)
model3 = BernoulliNB()
model4 = GaussianNB()

from sklearn.grid_search import GridSearchCV

param_grid = { 
    'alpha': [.001, .01, .1, 1],
}
CV = GridSearchCV(estimator=model2, param_grid=param_grid)
CV.fit(X_train[:1000], y_train[:1000])
print(CV.best_params_)

# from sklearn.svm import SVC
# classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)
model2.fit(X_train, y_train) # 84.97%
model3.fit(X_train, y_train) # 82%
model4.fit(X_train, y_train) # 79%

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_nb = model2.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred_nb)
score = accuracy_score(y_test, y_pred_nb)
