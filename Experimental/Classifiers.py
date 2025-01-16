import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier


def run_svc_classifier(X_train, y_train, X_test, y_test):
    # vectorization
    vectorizer = TfidfVectorizer(
        max_features=6000, decode_error='replace', encoding='utf-8')
    vectorizer.fit(X_train['text'].values.astype('U'))
    pre_processed_X = vectorizer.fit_transform(
        X_train['text'].values.astype('U')).toarray()  # independent
    X_train = pd.DataFrame(
        pre_processed_X, columns=vectorizer.get_feature_names_out())

    # fitting model
    pac = svm.SVC()
    pac.fit(X_train, y_train)
    best_pac = pac

    # testing phase

    # Joining attributes as text column
    X_test['text'] = X_test['subject'] + ' ' + X_test['description']
    X_test = X_test.drop(['subject', 'description'], axis=1)
    print(X_test)

    # apply same vectorizing to X_test data
    X_test = vectorizer.transform(X_test['text'].values.astype('U')).toarray()
    X_test = pd.DataFrame(X_test, columns=vectorizer.get_feature_names_out())

    # predict
    predictions = best_pac.predict(X_test)

    # report
    print(classification_report(predictions, y_test))
