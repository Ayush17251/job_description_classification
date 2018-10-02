'''
Here I am using Google's pre trained word2vec model 'GoogleNews-vectors-negative300.bin'.
'''

print("Here I am using Google's pre trained word2vec model 'GoogleNews-vectors-negative300.bin , Make Sure to have it in your current working directory")
import string
import re
import numpy as np
import pandas as pd
import json
import os
import nltk
import warnings
warnings.filterwarnings("ignore")
import enchant 
eng_dict = enchant.Dict("en_US")
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from HTMLParser import HTMLParser
from gensim.models import KeyedVectors
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from sklearn.preprocessing import LabelEncoder


pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', -1)
df = pd.read_csv('data/document_departments.csv')

kf = pd.DataFrame(columns=['Description','Document ID'])
path_to_jsonfiles = 'data/docs'


'''
Case1 : Where Data is Missing for the job Description for approximately 400 files. I am using other data available
inside the Json files.
        --> JOB Keywords
        --> JOB Industry
        --> JOB Title
        --> Department
        --> Industry

Case2: Where data is available for Job Description. Using that record only.

'''


i = 0
for file in os.listdir(path_to_jsonfiles):
    full_filename = "%s/%s" % (path_to_jsonfiles, file)
    with open(full_filename) as f:
        data = json.load(f)
        if data["jd_information"]["description"] == '':
            description = ''.join(data['api_data']['job_keywords']).encode('ascii','ignore')
            description2 = ''.join(data['api_data']['job_industry']).encode('ascii','ignore')
            description3 = ''.join(data['api_data']['job_title']).encode('ascii','ignore')
            description5 = ''.join(data['other_details']['Department:']).encode('ascii','ignore')
            description6 = ''.join(data['other_details']['Industry:']).encode('ascii','ignore')
            jd_info = description +' '+description2 +' '+description3+' '+description5+' '+description6
            kf.loc[i,'Description'] =  jd_info
            kf.loc[i,'Document ID']= data['_id'].encode('ascii')
        else:
            kf.loc[i,'Description'] =  data["jd_information"]["description"]
            kf.loc[i,'Document ID']= data['_id'].encode('ascii')
        i = i+1
        
kf['Document ID'] = kf['Document ID'].astype(np.int64)

## Mapping the Department with its code
xf = df.merge(kf, on="Document ID")

## Removing HTML Secial Character
parser = HTMLParser()
xf['Description'] = xf['Description'].apply(lambda x : parser.unescape(x))


'''
Case1 : Preprocessing -> clean_text() --> removing all stop words and emails and phone numbers from the Job Description.

'''

def clean_text(text):
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    all_stops = stops | set(string.punctuation)
    text = [w for w in text if not w in all_stops]
    text = " ".join(text)
    text = re.sub(r'\S*@\S*\s?'," ",text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", "  ", text)
    text = re.sub(r"\-", "  ", text)
    text = re.sub(r"\=", "  ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"e - mail", "email", text)
    text = text.split()
    lemmatiser = WordNetLemmatizer()
    stemmed_words = [lemmatiser.lemmatize(word) for word in text]
    stemmed_words = [word.encode('ascii') for word in text]
    return " ".join(stemmed_words)

'''
Case2 : Splitting the words which are combined together using enchant module python.
        --> Example: segment_str(Welcomelocation) 
                     output: ['Welcome','location']
                    
'''

def segment_str(chars, exclude=None):
    words = []
    if not exclude:
        exclude = set()

    working_chars = chars
    while working_chars:
        for i in range(len(working_chars), 1, -1):
            segment = working_chars[:i]
            if eng_dict.check(segment) and segment not in exclude:
                words.append(segment)
                working_chars = working_chars[i:]
                break
        else:  
            if words:
                exclude.add(words[-1])
                return segment_str(chars, exclude=exclude)
            return [chars]
    return words

def remove_join(text):
    x =[]
    sent_tokenize_list = text.split(' ')
    s = [segment_str(chars) for chars in sent_tokenize_list]
    x.append(sum(s, []))
    return ' '.join(sum(x, []))


xf['Description'] = xf['Description'].apply(clean_text)
xf['Description'] = xf['Description'].apply(remove_join)



## Splitting the data for test and train after randomising it.

xf = xf.sample(frac=1)
train_data = xf[:900]
test_data = xf[900:]
df = train_data

x = len(df['Department'].unique())

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 559
EMBEDDING_DIM = 300
EMBEDDING_FILE = "../GoogleNews-vectors-negative300.bin"

## Tokenizing and ppadding the data
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df['Description'])
description_sequence = tokenizer.texts_to_sequences(df['Description'])
description_data = pad_sequences(description_sequence,MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index

## Encoding the output labels
le = LabelEncoder()
df['target'] = le.fit_transform(df['Department'])
category = to_categorical(df['target'])
data = description_data



VALIDATION_SPLIT = 0.4
indices = np.arange(data.shape[0]) # get sequence of row index
np.random.shuffle(indices) # shuffle the row indexes
data = data[indices] # shuffle data/product-titles/x-axis
category = category[indices] # shuffle labels/category/y-axis
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = category[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = category[-nb_validation_samples:]


word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)

embedding_layer = Embedding(embedding_matrix.shape[0], 
                            embedding_matrix.shape[1], 
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

## Creating the Model

model_1 = Sequential()
model_1.add(embedding_layer)
model_1.add(Conv1D(300,3,padding='valid',activation='relu',strides=1))
model_1.add(GlobalMaxPooling1D())
model_1.add(Dense(250))
model_1.add(Dropout(0.2))
model_1.add(Activation('relu'))
model_1.add(Dense(x))
model_1.add(Activation('softmax'))
model_1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model_1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=8, batch_size=40)
score = model_1.evaluate(x_val, y_val, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

y_pred = []
for i in test_data['Description']:
    example_product = i
    example_product = clean_text(example_product)
    example_product = remove_join(example_product)
    example_sequence = tokenizer.texts_to_sequences([example_product])
    example_padded_sequence = pad_sequences(example_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    y_pred.append(le.inverse_transform(model_1.predict_classes(example_padded_sequence, verbose=0)[0]))
    


y_actual = []
for i in test_data['Department']:
    y_actual.append(i)


from sklearn.metrics import accuracy_score

print('Accuracy on test data is '+str(100*accuracy_score(y_actual, y_pred)))