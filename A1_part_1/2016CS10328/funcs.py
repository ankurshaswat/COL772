import json
import string
import re

# import numpy as np
# from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,confusion_matrix
# from nltk.stem.snowball import SnowballStemmer
from nltk import wordpunct_tokenize
# stemmer = SnowballStemmer("english", ignore_stopwords=True)
translator = str.maketrans("","", string.punctuation)

# def get_metrics_from_pred(y_pred,y_true):
#     mse = mean_squared_error(y_pred,y_true)
    
#     try:
#         f1_scor = f1_score(y_true, y_pred, average='weighted')
#         acc = accuracy_score(y_true, y_pred)
      
#     except:
#         y_pred = np.round(y_pred)
        
#         f1_scor = f1_score(y_true, y_pred, average='weighted')
#         acc = accuracy_score(y_true, y_pred)
        
#     print("MSE = ",mse," F1 = ",f1_scor," Accuracy = ",acc)

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

def read_test_file(path):
    data_X = []
    with open(path, 'r') as data_file:
        line = data_file.readline()
        while line:
            data = json.loads(line)
            data_X.append(data['review'])
            line = data_file.readline()
    return data_X

def remove_repeats(sentence):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", sentence)

def tokenizer1(sentence):
    # sentence = sentence.translate(translator)      # Remove punctuations
    # sentence = sentence.lower()                    # Convert to lowercase
    #sentence = re.sub(r'\d+', '', sentence)        # Remove Numbers
    sentence = remove_repeats(sentence)            # Remove repeated characters
#     sentence = sentence.strip()                    # Remove Whitespaces
    tokens = wordpunct_tokenize(sentence)          # Tokenize
#     tokens = word_tokenize(sentence)          # Tokenize
    
    # for i in range(len(tokens)):                    # Stem word
        # tokens[i] = stemmer.stem(tokens[i])
    return tokens

def predict(model_123_45,model_4_5,model_12_3,model_1_2,X_counts):
    pred_123_45 = model_123_45.predict_proba(X_counts)
    pred_12_3 = model_12_3.predict_proba(X_counts)
    pred_1_2 = model_1_2.predict_proba(X_counts)
    pred_4_5 = model_4_5.predict_proba(X_counts)

    pred = []

    for i in range(len(pred_123_45)):
        pred.append(pred_123_45[i][0]*pred_12_3[i][0]*pred_1_2[i][0]*1.0+
                    pred_123_45[i][0]*pred_12_3[i][0]*pred_1_2[i][1]*2.0+
                    pred_123_45[i][0]*pred_12_3[i][1]*3.0+
                    pred_123_45[i][1]*pred_4_5[i][0]*4.0+
                    pred_123_45[i][1]*pred_4_5[i][1]*5.0)

    return pred
