from nltk.stem import WordNetLemmatizer
import time
import gensim as gensim
import pandas as pd
import warnings
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.options.display.max_colwidth = 1000

data_path = "data//Code review w_ keyword 'refactor'  (Spring 2023).xlsx"


def load_data():
    data = pd.read_excel(open(data_path, 'rb'),sheet_name='Manual labeling', usecols="A:E")
    return data

def pre_process(X):
    data = X
    clean_data = data
    clean_data['subject'] = clean_data['subject'].fillna(
        "no")  # can use " " instead of unknown for improvement
    clean_data['description'] = clean_data['description'].fillna(
        "no")  # can use " " instead of unknown for nothing
    clean_data.fillna('', inplace=True)
    # Remove Stopwords, Punctuation and Special Characters
    clean_data['subject'] = clean_data['subject'].str.lower()
    clean_data['description'] = clean_data['description'].str.lower()

    clean_data['subject'] = clean_data['subject'].apply(
        lambda x: gensim.parsing.preprocessing.remove_stopwords("".join(x)))
    clean_data['description'] = clean_data['description'].apply(
        lambda x: gensim.parsing.preprocessing.remove_stopwords("".join(x)))

    clean_data['subject'] = clean_data['subject'].str.replace(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')  # Remove stop words
    clean_data['description'] = clean_data['description'].str.replace(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')

    clean_data['subject'] = clean_data['subject'].str.replace(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')  # Remove unwanted characters
    clean_data['description'] = clean_data['description'].str.replace(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')

    clean_data['subject'] = clean_data['subject'].str.replace(r'[\'-]', '')
    clean_data['description'] = clean_data['description'].str.replace(
        r'[\'-]', '')

    clean_data['subject'] = clean_data['subject'].str.replace(
        r'[\'\",()*&^%$#@!~`+=|/<>?{}\[\]\/\\:;\_]]', ' ')  # remove punctuation
    clean_data['description'] = clean_data['description'].str.replace(
        r'[\'\",()*&^%$#@!~`+=|/<>?{}\[\]\/\\:;\_]]', ' ')

    clean_data['subject'] = clean_data['subject'].apply(lambda x: ' '.join(
        [word for word in x.split() if not word.startswith('url')]))  # remove urls (mostly stackoverflow)
    clean_data['description'] = clean_data['description'].apply(
        lambda x: ' '.join([word for word in x.split() if not word.startswith('url')]))

    # clean_data['subject'] = clean_data['subject'].apply(lambda x: ' '.join([word for word in x.split(' ') if len(word) < 25]))
    # clean_data['description'] = clean_data['description'].apply(lambda x: ' '.join([word for word in x.split(' ') if len(word) < 25]))

    # clean_data['text'] = clean_data['text'].map(lambda x: re.sub(r'\W+', ' ', x))

    clean_data['subject'] = clean_data['subject'].str.replace(
        r'[0-9]', '')  # get rid of numbers
    clean_data['description'] = clean_data['description'].str.replace(
        r'[0-9]', '')

    clean_data['subject'] = clean_data['subject'].str.replace(r'[^a-z]', ' ')
    clean_data['description'] = clean_data['description'].str.replace(
        r'[^a-z]', ' ')    # get rid of any non english characters

    # Remove Change-ID:
    clean_data['description'] = clean_data['description'].str.split(
        'changeid').str[0]
    clean_data['description'] = clean_data['description'].str.replace(
        '   ', ' ')  # Remove three white spaces
    clean_data['description'] = clean_data['description'].str.replace(
        '  ', ' ')  # Alter two white spaces to one

    clean_data['subject'] = clean_data['subject'].str.replace(r'[^a-z]', ' ')
    clean_data['description'] = clean_data['description'].str.replace(
        r'[^a-z]', ' ')

    clean_data['description'] = clean_data['description'].str.replace(
        r'\b\w\b', '').str.replace(r'\s+', ' ')  # Remove single characters

    # testing
    # clean_data['description'] = clean_data['description'].str.replace('py',"")  # Remove py(extension)
    # try not removing bug number
    # Bug number as a separate attribute (figure out what bug number is..)
    # Data is bad!!! (same X for different Y) see excel red highlight
    # Researchers have tried to balance the data and messed up!!

    clean_data['description'] = clean_data['description'].replace(
        r'^\s*$', 'no description', regex=True)

    clean_data['description'] = clean_data['description'].apply(
        lemmatize_words)

    # Joining attributes as text column
    clean_data['text'] = clean_data['subject'] + \
        ' ' + clean_data['description']

    return clean_data


def lemmatize_words(text):
    # apply lemmatization
    lemmatizer = WordNetLemmatizer()

    words = text.split()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]
    return ' '.join(words)
