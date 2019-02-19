import json
import re
import string
import scipy

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm_notebook as tqdm

from nltk import wordpunct_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.linear_model import LinearRegression,SGDClassifier,ElasticNet,LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

train_path = 'data/train.json'
dev_path = 'data/dev.json'

translator = str.maketrans("","", string.punctuation)
stemmer = SnowballStemmer("english", ignore_stopwords=True)

def read_file(path):
    data_X = []
    data_Y = []
    with open(path, 'r') as data_file:
        line = data_file.readline()
        while line:
            data = json.loads(line)
            data_X.append(data['review'])
            data_Y.append(data['ratings'])
            line = data_file.readline()
    return data_X,data_Y

def get_metrics_from_pred(y_pred,y_true):
    mse = mean_squared_error(y_pred,y_true)
    
    try:
        f1_scor = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true,y_pred)
      
    except:
        y_pred = np.round(y_pred)
        
        f1_scor = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true,y_pred)
        
    print("MSE = ",mse," F1 = ",f1_scor," Accuracy = ",acc)
    plt.matshow(conf_matrix)
    plt.colorbar()
    
def get_metrics(model,X,y_true):
    y_pred = model.predict(X)
    get_metrics_from_pred(y_pred,y_true)
    
def get_metrics_using_probs(model,X,y_true):
    y_pred = model.predict_proba(X)
    y_pred = np.average(y_pred,axis=1, weights=[1,2,3,4,5])*15
    get_metrics_from_pred(y_pred,y_true)
    
def remove_repeats(sentence):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", sentence)

def tokenizer1(sentence):
    sentence = sentence.translate(translator)      # Remove punctuations
    sentence = sentence.lower()                    # Convert to lowercase
    sentence = re.sub(r'\d+', '', sentence)        # Remove Numbers
    sentence = remove_repeats(sentence)            # Remove repeated characters
#     sentence = sentence.strip()                    # Remove Whitespaces
    tokens = wordpunct_tokenize(sentence)          # Tokenize
#     tokens = word_tokenize(sentence)          # Tokenize
    
#     for i in range(len(tokens)):                    # Stem word
#         tokens[i] = stemmer.stem(tokens[i])
    return tokens

tokenize = tokenizer1

X_train,Y_train = read_file(train_path)
X_dev,Y_dev = read_file(dev_path)

# X_train = X_train[0:100]
# Y_train = Y_train[0:100]
# X_dev = X_dev[0:100]
# Y_dev = Y_dev[0:100]

processed_stopwords = []

for word in stopwords.words('english'):
    processed_stopwords += tokenize(word)
    
# vectorizer = TfidfVectorizer(strip_accents='ascii',
#                              lowercase=True,
#                              tokenizer=tokenize,
#                              stop_words=processed_stopwords,
#                              ngram_range=(1,1),
#                              binary=True,
#                              norm='l2',
#                              analyzer='word')

# vectorizer = TfidfVectorizer(binary=True,tokenizer=tokenize)

# vectorizer = TfidfVectorizer(tokenizer=tokenize)

# vectorizer = TfidfVectorizer(tokenizer=tokenize,ngram_range=(1,2),binary=True)

vectorizer = CountVectorizer(tokenizer=tokenize,ngram_range=(1,2))

X_train_counts = vectorizer.fit_transform(X_train)
X_dev_counts = vectorizer.transform(X_dev)

print("vectorizer done")

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

model_123_45 = LogisticRegression(verbose=1,solver='sag')
model_123_45.fit(X_train_counts,Y_modified)

model_4_5 = LogisticRegression(verbose=1,solver='sag')
model_4_5.fit(X_train_counts_4_5,Y_train_4_5)

model_12_3 = LogisticRegression(verbose=1,solver='sag')
model_12_3.fit(X_train_counts_12_3,Y_train_12_3)

model_1_2 = LogisticRegression(verbose=1,solver='sag')
model_1_2.fit(X_train_counts_1_2,Y_train_1_2)

pred_123_45 = model_123_45.predict_proba(X_dev_counts)
pred_12_3 = model_12_3.predict_proba(X_dev_counts)
pred_1_2 = model_1_2.predict_proba(X_dev_counts)
pred_4_5 = model_4_5.predict_proba(X_dev_counts)

pred = []

for i in tqdm(range(len(pred_123_45))):
    pred.append(pred_123_45[i][0]*pred_12_3[i][0]*pred_1_2[i][0]*1.0+
                pred_123_45[i][0]*pred_12_3[i][0]*pred_1_2[i][1]*2.0+
                pred_123_45[i][0]*pred_12_3[i][1]*3.0+
                pred_123_45[i][1]*pred_4_5[i][0]*4.0+
                pred_123_45[i][1]*pred_4_5[i][1]*5.0)
get_metrics_from_pred(pred,Y_dev)