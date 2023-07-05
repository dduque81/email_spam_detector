import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
import re
from bs4 import BeautifulSoup

#df = pd.read_csv('datasets/email/csv/spam-apache.csv', names = ['clase', 'contenido'])
stopwords = set(stopwords.words('english'))

def clean_data(df):
    """This function clean and prepare the dataset"""
    def strip_html(text):
        """Remover etiquetas html"""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()
    
    def remove_between_square_brackets(text):
        """Eliminar corchetes"""
        return re.sub('\[[^]]*\]', '', text)
    
    def denoise_text(text):
        """Ejecuta las dos funciones anteriores de limpieza"""
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        return text
    
    def remove_emails(text):
        """Elimina los emails"""
        pattern = r'\S+@\S+'
        text = re.sub(pattern, '', text)
        return text
    
    def remove_numbers(text):
        """
        Eliminación de números
        """
        pattern = r'\d+'
        text = re.sub(pattern, '', text)
        return text
    
    def remove_special_characters(text, remove_digits=True):
        """Eliminar caracteres especiales"""
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern, '', text)
        return text
    
    def lowercase_text(text):
        """Texto a minuscula"""
        return text.lower()
    
    def clean_text(text):
        """Eliminar caracteres saltos de linea (\n)"""
        text = re.sub(r'\n', ' ', text)
        
        """Eliminar espacios multiples y dejar un unico espacio"""
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def contenido_cleaning(df):
        """Limpieza de contenido de emails"""
        df['contenido_limpio'] = df['contenido'].apply(denoise_text)
        df['contenido_limpio'] = df['contenido_limpio'].apply(remove_emails)
        df['contenido_limpio'] = df['contenido_limpio'].apply(remove_numbers)
        df['contenido_limpio'] = df['contenido_limpio'].apply(remove_special_characters)
        df['contenido_limpio'] = df['contenido_limpio'].apply(lowercase_text)
        df['contenido_limpio'] = df['contenido_limpio'].apply(clean_text)
        
        return df
    
    df_n = contenido_cleaning(df)
    return df_n

def tokenize_data(df):
    """This function tokenize the strings"""
    df['tokens'] = df['contenido_limpio'].apply(lambda x: word_tokenize(x))
    return df

def lemmatize_tokens(tokens):
    """This function lemmatize the tokens"""
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def remove_stopwords(tokens):
    """This function removes stopwords from tokens"""
    tokens_no_stopwords = []
    for token in tokens:
        if token not in stopwords:
            tokens_no_stopwords.append(token)
            
    return tokens_no_stopwords

def process_data(df):
    """This function execute the other functions cleaning and preparing the data"""
    df = clean_data(df)
    df = tokenize_data(df)
    df['lemmatized_tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['tokens_no_stopwords'] = df['lemmatized_tokens'].apply(remove_stopwords)
    print('BEFORE LEMMATIZING\n')
    print(df['tokens'][0][:100])
    print('AFTER LEMMATIZING\n')
    print(df['lemmatized_tokens'][0][:100])
    return df

#df_processed = process_data(df)