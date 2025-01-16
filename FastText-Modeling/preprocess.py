import math
import time
import gensim as gensim
import pandas as pd
import numpy as np
import warnings
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 10)
#pd.set_option('display.max_columns', 1000)
pd.options.display.max_colwidth = 1000

def pre_process( X):
    data = X
    clean_data = data
    clean_data['subject'] = clean_data['subject'].fillna("")  # can use " " instead of unknown for improvement
    clean_data['description'] = clean_data['description'].fillna("")  # can use " " instead of unknown for nothing
    clean_data.fillna('', inplace=True)
    # Remove Stopwords, Punctuation and Special Characters
    clean_data['subject'] = clean_data['subject'].str.lower()
    clean_data['description'] = clean_data['description'].str.lower()

    clean_data['subject'] = clean_data['subject'].apply(lambda x: gensim.parsing.preprocessing.remove_stopwords("".join(x)))
    clean_data['description'] = clean_data['description'].apply(lambda x: gensim.parsing.preprocessing.remove_stopwords("".join(x)))

    clean_data['subject'] = clean_data['subject'].str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '') # Remove stop words
    clean_data['description'] = clean_data['description'].str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')
    """
Directly below here looks like duplicate code? 
So I commented it out
    """
    #clean_data['subject'] = clean_data['subject'].str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '') # Remove unwanted characters
    #clean_data['description'] = clean_data['description'].str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')

    clean_data['subject'] = clean_data['subject'].str.replace(r'[\'-]', '')
    clean_data['description'] = clean_data['description'].str.replace(r'[\'-]', '')

    clean_data['subject'] = clean_data['subject'].str.replace(r'[\'\",()*&^%$#@!~`+=|/<>?{}\[\]\/\\:;\_]]',' ')  # remove punctuation
    clean_data['description'] = clean_data['description'].str.replace(r'[\'\",()*&^%$#@!~`+=|/<>?{}\[\]\/\\:;\_]]', ' ')

    clean_data['subject'] = clean_data['subject'].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('url')])) # remove urls (mostly stackoverflow)
    clean_data['description'] = clean_data['description'].apply( lambda x: ' '.join([word for word in x.split() if not word.startswith('url')]))

    # clean_data['subject'] = clean_data['subject'].apply(lambda x: ' '.join([word for word in x.split(' ') if len(word) < 25]))
    # clean_data['description'] = clean_data['description'].apply(lambda x: ' '.join([word for word in x.split(' ') if len(word) < 25]))

    # clean_data['text'] = clean_data['text'].map(lambda x: re.sub(r'\W+', ' ', x))

    clean_data['subject'] = clean_data['subject'].str.replace(r'[0-9]', '')  # get rid of numbers
    clean_data['description'] = clean_data['description'].str.replace(r'[0-9]', '')

    clean_data['subject'] = clean_data['subject'].str.replace(r'[^a-z]', ' ')
    clean_data['description'] = clean_data['description'].str.replace(r'[^a-z]', ' ')    # get rid of any non english characters

    #clean_data['description'] = clean_data['description'].str.split('changeid').str[0]  # Remove Change-ID:
    #This line is broken and just deletes the entire description. so I commented out. Change id is removed by 
    # the lemmatizer or somewhere else anyways
    clean_data['description'] = clean_data['description'].str.replace('   ', ' ') # Remove three white spaces
    clean_data['description'] = clean_data['description'].str.replace('  ', ' ')# Alter two white spaces to one

    clean_data['subject'] = clean_data['subject'].str.replace(r'[^a-z]', ' ')
    clean_data['description'] = clean_data['description'].str.replace(r'[^a-z]', ' ')

    clean_data['description'] = clean_data['description'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ') # Remove single characters

    #
    # testing
    #clean_data['description'] = clean_data['description'].str.replace('py',"")  # Remove py(extension)
    # try not removing bug number 
    # Bug number as a separate attribute (figure out what bug number is..)
    # Data is bad!!! (same X for different Y) see excel red highlight
    # Researchers have tried to balance the data and messed up!!
    #SEE BELOW
    #SEE BELOW
    #SEE BELOW
    # RESPONSE - this is multi class not from balancing, some have multiple labels

    clean_data['description'] = clean_data['description'].replace(r'^\s*$', 'no description', regex=True)

    #apply lemmatization
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(text):
        words = text.split()
        words = [lemmatizer.lemmatize(word, pos='v') for word in words]
        return ' '.join(words)
    clean_data['description'] = clean_data['description'].apply(lemmatize_words)
    clean_data['subject']= clean_data['subject'].apply(lemmatize_words)
    return clean_data

def split_train(data):
    #dropping duplicates
    #data = data.drop_duplicates()
    # 0.8 train & 0.2 test
    y = data['Category']
    X = data.drop(['Category'], axis=1)


    # Preprocess X
    X_processed = pre_process(X)

    #join X and Y
    data = pd.concat([X_processed, y], axis=1)