import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

def machine_learning_models(df):
    """This function builds and evaluate the models"""
    X = df['tokens_no_stopwords'].apply(' '.join)
    y = df['clase']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
    vectorizer = CountVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    def log_reg_model(X_train_vect, X_test_vect, y_train, y_test):
        """Logistic regression model"""
        log_reg = LogisticRegression()
        log_reg.fit(X_train_vect, y_train)
        log_reg_pred = log_reg.predict(X_test_vect)
        print('REGRESIÓN LOGISTICA: \n')
        print(confusion_matrix(y_test, log_reg_pred))
        print(classification_report(y_test, log_reg_pred))

    def dec_tree_model(X_train_vect, X_test_vect, y_train, y_test):
        """Decision tree model"""
        tree = DecisionTreeClassifier()
        tree.fit(X_train_vect, y_train)
        tree_pred = tree.predict(X_test_vect)
        print('ARBOLES DE DECISIÓN: \n')
        print(confusion_matrix(y_test, tree_pred))
        print(classification_report(y_test, tree_pred))

    def naive_bayes_model(X_train_vect, X_test_vect, y_train, y_test):
        """Naive bayes model"""
        nb = MultinomialNB()
        nb.fit(X_train_vect, y_train)
        nb_pred = nb.predict(X_test_vect)
        print('ARBOLES DE DECISIÓN: \n')
        print(confusion_matrix(y_test, nb_pred))
        print(classification_report(y_test, nb_pred))

    log_reg_model(X_train_vect, X_test_vect, y_train, y_test)
    dec_tree_model(X_train_vect, X_test_vect, y_train, y_test)
    naive_bayes_model(X_train_vect, X_test_vect, y_train, y_test)