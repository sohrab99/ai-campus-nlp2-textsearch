#!/usr/bin/env python3
import os
import tempfile
import nltk
import re
import numpy as np
import pandas as pd
from pprint import pprint
import pickle
import gensim
import gensim.corpora as corpora
import gensim.models as models
import gensim.similarities as similarities
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import get_tmpfile
from gensim.models import LsiModel
# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from','tel', 'mail', 'zuma', 'using', 'image' ,'subject', 're', 'edu', 'use', 'information', 'please', 'contact', 'email', 'eyevine'])
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
# warnings
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
TEMP_FOLDER = tempfile.gettempdir()

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


df = pd.read_json('image_meta_data.json')
df.set_index('image_index', inplace=True)
#print(df.content.unique())
print(df.head(5))
df.to_pickle(os.path.join(TEMP_FOLDER, 'image_meta_data_500.pkl')) 
# Convert to list
data = df.content.values.tolist()
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
data = [re.sub('\S*.com', '', sent) for sent in data]
#print(data[:1])
# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]
# Convert to list
data_words = list(sent_to_words(data))
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
#lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
id2word.save(os.path.join(TEMP_FOLDER, 'nlp2_img_data.dict'))  # store the dictionary, for future reference
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
#corpus = [id2word.doc2bow(data_lemmatized) for data_lemmatized in data_lemmatized]
#print(corpus[:5])
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'nlp2_img_corpus.mm'), corpus)  # store to disk, for later use
#corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'nlp2_img_corpus.mm'))
#print(corpus)
lsi = models.LsiModel(corpus, id2word=id2word, num_topics=100)
tmp_fname = get_tmpfile("nlp2_img_lsi.model")
lsi.save(tmp_fname)  # save model
print(lsi)
index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to lsi space and index it
#print(index)
index.save(os.path.join(TEMP_FOLDER,'nlp2_img_ind.index'))
#print(id2word.token2id)
#print(list(index))
print("model saved")

































