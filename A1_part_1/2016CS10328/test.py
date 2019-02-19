import pickle
import sys

from funcs import *

pickle_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

dbfile = open(pickle_path, 'rb')      
db = pickle.load(dbfile)

vectorizer =     db['vectorizer']  
model_123_45 =   db['model_123_45']  
model_4_5 =  db['model_4_5']  
model_12_3 =     db['model_12_3']  
model_1_2 =  db['model_1_2']  

dbfile.close()

X = read_test_file(test_path)
# X,Y = read_file(test_path)

X_counts = vectorizer.transform(X)

predictions = predict(model_123_45,model_4_5,model_12_3,model_1_2,X_counts)

# get_metrics_from_pred(predictions,Y)


with open(output_path, 'w') as file:
    for prediction in predictions:
        file.write(str(prediction)+'\n')