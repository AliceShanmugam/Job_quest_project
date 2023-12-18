# Module to preprocess the data before fitting the ML model
# Vectorizer and label encoding.
# Finally, split the df into X_train, y_train, X_val, y_val with Stratify( )

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from scripts.preprocessor.preprocessing_dataset import package_category_df, package_job_title_df
from scripts.preprocessor.preprocessing_dataset import remove_stopwords, clean_description, remove_title
from scripts.preprocessor.preprocessing_dataset import analyze_and_clean_title, standardize_job_title, replace_title_with_target, group_titles

""" SCRIPT PREPROC ML :
etape 1 : train-test split
etape 2 : encoder les labels, store l'encoder
etape 3 : vectoriser les X, store le vectorizer
Retourne: X_train_transformed, X_test, y_train, y_test
"""

def train_test_split_df(df):
    """Train-test split the dataframe : test size is 0.2, stratify to ensure equal
    proportions of classes in train and test, fix random state.
    Returns X_train, X_test, y_train, y_test.
    """

    X = df['description']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test


def vectorize_X(X_train, X_test):
    """ Vectorizer : use CountVectorizer and vectorize X.
    Returns X_train_transformed and x_test_transformed
    """
    vectorizer = CountVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    return X_train_transformed, X_test_transformed, vectorizer


def target_encode(y_train, y_test):
    """ Target encode : label encoding of categories.
    returns y_train_transformed, y_test_transformed and fitted label encoder
    """
    label_encoder =  LabelEncoder()
    y_train_transformed = label_encoder.fit_transform(y_train)
    y_test_transformed = label_encoder.transform(y_test)
    return y_train_transformed, y_test_transformed, label_encoder

if __name__ == '__main__' :
    print('ML Preprocessing function')
