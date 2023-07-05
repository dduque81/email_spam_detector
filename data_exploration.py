import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wordcloud
import nltk
from nltk import FreqDist, bigrams
from nltk.util import ngrams
from nltk.collocations import *

def plots(df):
    """This function plot and explore the data"""

    def count_plot(df):
        """Distribution of the class"""
        counts = df['clase'].value_counts()

        plt.bar(counts.index, counts.values)
        plt.xlabel('Clase')
        plt.xticks([-1, 1], ['Spam', 'No spam'])
        plt.ylabel('Proporción')
        plt.title('Distribución')
        plt.show()

    def hist(df):
        """This function shows the histogram of number of words per email"""
        df['len_words'] = df['contenido_limpio'].apply(lambda x: len(x.split()))
        df_neg = df[df['clase'] == -1]
        df_pos = df[df['clase'] == 1]
        # Crear subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Histograma de longitud de palabras para negativos
        ax1.hist(df_neg['len_words'], bins=20, edgecolor='k', alpha=0.7)
        ax1.set_title('Histograma de Longitud de Palabras - SPAM')
        ax1.set_xlabel('Longitud de Palabras')
        ax1.set_ylabel('Frecuencia')

        # Histograma de longitud de palabras para positivos
        ax2.hist(df_pos['len_words'], bins=20, edgecolor='k', alpha=0.7)
        ax2.set_title('Histograma de Longitud de Palabras - NO SPAM')
        ax2.set_xlabel('Longitud de Palabras')
        ax2.set_ylabel('Frecuencia')

        # Ajustar espaciado entre subplots
        plt.tight_layout()

        # Mostrar los histogramas
        plt.show()

        return(df_neg, df_pos)
    
    def find_ngrams(df):
        """This function finds bigrams and trigrams"""
        #POSITIVES
        #BIGRAMS
        no_spam_tokens = []
        for tokens in df_pos['tokens_no_stopwords']:
            for token in tokens:
                no_spam_tokens.append(token)

        bigrams = list(ngrams(no_spam_tokens, 2))
        pos_bi_freq_dist = FreqDist(bigrams)
        pos_top_10_bigrams = pos_bi_freq_dist.most_common(10)
        pos_bigram_labels = [', '.join(bigram) for bigram, freq in pos_top_10_bigrams]
        pos_bigram_frequencies = [freq for bigram, freq in pos_top_10_bigrams]

        #TRIGRAMS
        trigrams = list(ngrams(no_spam_tokens, 3))
        pos_tri_freq_dist = FreqDist(trigrams)
        pos_top_10_trigrams = pos_tri_freq_dist.most_common(10)
        pos_trigram_labels = [', '.join(trigram) for trigram, freq in pos_top_10_trigrams]
        pos_trigram_frequencies = [freq for trigram, freq in pos_top_10_trigrams]

        #NEGATIVES
        #BIGRAMS
        spam_tokens = []
        for tokens in df_neg['tokens_no_stopwords']:
            for token in tokens:
                spam_tokens.append(token)

        bigrams = list(ngrams(spam_tokens, 2))
        neg_bi_freq_dist = FreqDist(bigrams)
        neg_top_10_bigrams = neg_bi_freq_dist.most_common(10)
        neg_bigram_labels = [', '.join(bigram) for bigram, freq in neg_top_10_bigrams]
        neg_bigram_frequencies = [freq for bigram, freq in neg_top_10_bigrams]

        trigrams = list(ngrams(spam_tokens, 3))
        neg_tri_freq_dist = FreqDist(trigrams)
        neg_top_10_trigrams = neg_tri_freq_dist.most_common(10)
        neg_trigram_labels = [', '.join(trigram) for trigram, freq in neg_top_10_trigrams]
        neg_trigram_frequencies = [freq for trigram, freq in neg_top_10_trigrams]

        # Crear subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

        # Plot para la Figura 1
        ax1.barh(range(10), pos_bigram_frequencies)
        ax1.set_yticks(range(10))
        ax1.set_yticklabels(pos_bigram_labels, rotation = 45)
        ax1.set_xlabel('Frecuencia')
        ax1.set_ylabel('Bigramas')
        ax1.set_title('Top 10 Bigramas Frecuentes - NO Spam')
        # Plot para la Figura 2
        ax2.barh(range(10), neg_bigram_frequencies)
        ax2.set_yticks(range(10))
        ax2.set_yticklabels(neg_bigram_labels, rotation = 45)
        ax2.set_xlabel('Frecuencia')
        #ax2.set_ylabel('Bigramas')
        ax2.set_title('Top 10 Bigramas Frecuentes - Spam')
        # Plot para la Figura 3
        ax3.barh(range(10), pos_trigram_frequencies)
        ax3.set_yticks(range(10))
        ax3.set_yticklabels(pos_trigram_labels, rotation = 45)
        ax3.set_xlabel('Frecuencia')
        ax3.set_ylabel('Trigramas')
        ax3.set_title('Top 10 Trigramas Frecuentes - NO Spam')
        # Plot para la Figura 4
        ax4.barh(range(10), neg_trigram_frequencies)
        ax4.set_yticks(range(10))
        ax4.set_yticklabels(neg_trigram_labels, rotation = 45)
        ax4.set_xlabel('Frecuencia')
        #ax4.set_ylabel('Trigramas')
        ax4.set_title('Top 10 Trigramas Frecuentes - Spam')

        return(no_spam_tokens, spam_tokens)

    def collocations(no_spam_tokens, spam_tokens):
        """This function finds the collocations of the emails"""
        no_spam_bigram_measure = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(no_spam_tokens)
        finder.apply_freq_filter(20)
        no_spam_collocations = finder.nbest(no_spam_bigram_measure.pmi, 10)

        spam_bigram_measure = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(spam_tokens)
        finder.apply_freq_filter(20)
        spam_collocations = finder.nbest(spam_bigram_measure.pmi, 10)

        print('No spam - Colocaciones\n')
        print(no_spam_collocations)
        print('\nSpam - Colocaciones\n')
        print(spam_collocations)

    def word_cloud(no_spam_tokens, spam_tokens):
        """This function plots the word cloud of the data"""
        #Defines the word cloud
        wc = wordcloud.WordCloud(width = 400, height = 400, max_words=70, background_color = 'white', contour_width = 1, 
                         contour_color='steelblue', min_font_size=10, max_font_size=300, prefer_horizontal=0.8, 
                                 colormap='viridis')
        #Subplot for spam and no spam word cloud
        # Crear la figura y los subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Generar el Word Cloud para spam
        wordcloud_words_spam = ' '.join(spam_tokens)
        wc.generate(wordcloud_words_spam)
        ax2.imshow(wc, interpolation='bilinear')
        ax2.axis('off')
        ax2.set_title('SPAM')

        # Generar el Word Cloud para no spam
        wordcloud_words_no_spam = ' '.join(no_spam_tokens)
        wc.generate(wordcloud_words_no_spam)
        ax1.imshow(wc, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title('NO SPAM')

        plt.tight_layout()
        plt.show()

    count_plot(df)
    df_neg, df_pos = hist(df)
    #ngrams(df)
    no_spam_tokens, spam_tokens = find_ngrams(df)
    collocations(no_spam_tokens, spam_tokens)
    word_cloud(no_spam_tokens, spam_tokens)