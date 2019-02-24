import json
import numpy as np
import pickle

from funcs import *

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import sys

train_path = sys.argv[1]
dev_path = sys.argv[2]
pickle_path = sys.argv[3]

tokenize = tokenizer1

X_t,Y_t = read_file(train_path)
X_d,Y_d = read_file(dev_path)

X_train = X_t + X_d
Y_train = Y_t + Y_d

vectorizer = TfidfVectorizer(tokenizer=tokenize,ngram_range=(1,2),binary=True)

X_train_counts = vectorizer.fit_transform(X_train)

indices = np.where(list(map(lambda x: x<=3,Y_train)))[0]
X_train_counts_12_3 = X_train_counts[indices]
Y_train_12_3 = [1 if Y_train[j]==3 else 0 for j in indices]

indices = np.where(list(map(lambda x:x>3,Y_train)))[0]
X_train_counts_4_5 = X_train_counts[indices]
Y_train_4_5 = [Y_train[j] for j in indices]

indices = np.where(list(map(lambda x:x<3,Y_train)))[0]
X_train_counts_1_2 = X_train_counts[indices]
Y_train_1_2 = [Y_train[j] for j in indices]

def modif(x):
    if (x>3):
        return 1
    else:
        return 0

Y_modified = list(map(lambda x: modif(x),Y_train))

model_123_45 = LogisticRegression(solver='sag')
model_123_45.fit(X_train_counts,Y_modified)

model_4_5 = LogisticRegression(solver='sag')
model_4_5.fit(X_train_counts_4_5,Y_train_4_5)

model_12_3 = LogisticRegression(solver='sag')
model_12_3.fit(X_train_counts_12_3,Y_train_12_3)

model_1_2 = LogisticRegression(solver='sag')
model_1_2.fit(X_train_counts_1_2,Y_train_1_2)

db = {} 
db['vectorizer'] = vectorizer 
db['model_123_45'] = model_123_45 
db['model_4_5'] = model_4_5 
db['model_12_3'] = model_12_3 
db['model_1_2'] = model_1_2 
    
dbfile = open(pickle_path, 'ab')
    
pickle.dump(db, dbfile)             
dbfile.close()
