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
import pyLDAvis.gensim  # don't skip this
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

app = Flask(__name__)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5577, debug=True)

@app.route('/', methods=['GET', 'POST'])

def main ():
    current_article = request.values.get('current_article', None)
    if current_article is None:
        return render_template('index.html', article=None)
    # process url
    # sample URL: https://www.cnbc.com/2019/01/08/chairman-eddie-lampert-to-get-another-chance-to-save-sears-sources-say.html
    print(current_article)
    query_text = current_article
    query_text = re.sub('\S*@\S*\s?', '', query_text)
    query_text = re.sub('\s+', ' ', query_text)
    query_text = re.sub('\S*.com', ' ', query_text)
    query_text = re.sub("\'", "", query_text)
    
    data_words = list(sent_to_words(query_text))
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #print(data_lemmatized[0])
	
    id2word = corpora.Dictionary.load(os.path.join(TEMP_FOLDER, 'nlp2_img_data.dict'))
    #print(id2word.token2id)
    corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'nlp2_img_corpus.mm'))
    lsi= LsiModel.load(os.path.join(TEMP_FOLDER, 'nlp2_img_lsi.model'))
    index = similarities.MatrixSimilarity.load(os.path.join(TEMP_FOLDER, 'nlp2_img_ind.index'))
    print(lsi)
    print(index)
    #print(list(index))
    #print(data_lemmatized[0])
    vec_bow = id2word.doc2bow(data_lemmatized[0])
    #print(vec_bow)
    vec_lsi = lsi[vec_bow]
    #print(vec_lsi)
    sims = index[vec_lsi]
    #print(list(enumerate(sims)))	
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    top_10_vectors = sims[:10]
    #print(top_10_vectors)
    a_doc_index = []
    a_prob_match = []
    image_url_display = ""
    a_image_url = []
    df = pd.read_pickle(os.path.join(TEMP_FOLDER, 'image_meta_data_500.pkl'))
    #df.set_index('image_index', inplace=True)
    print(df.head(10))
    x = 0
    for x in range(0, 9):
        s_vector = tuple(map(str, top_10_vectors[x]))
        # print(s_vector)
        img_index_perc = ','.join((s_vector))
        s_doc_index = int(img_index_perc[0:img_index_perc.find(',')])
        s_doc_perc = float(img_index_perc[img_index_perc.find(',') + 1:])
        # print(img_index_perc)
        # print(s_doc_index)
        # print(s_doc_perc)
        if s_doc_perc >= 0.65:
            a_doc_index.append(s_doc_index)
            a_prob_match.append(s_doc_perc)
    print(a_doc_index)
    print(a_prob_match)
    counter = 0
    dis_img_url = ""
    for each_index in a_doc_index:
        image_url_display = df.iloc[each_index, 2]
        a_image_url.append(df.iloc[each_index, 2])
        #df1 = df.loc[df['image_index'] == each_index]
        #df1 = df.loc[[each_index],['image_url']]
        #image_url_display = df1.iloc[0,0]
        # print (each_index)
        print("Image " + str(image_url_display) + " matches " + str(a_prob_match[counter]) + " with entered text")
        counter = counter + 1
    #return render_template('index.html',
    #                        article=current_article)
    return render_template('gallery.html',
                            image_url= a_image_url, article=current_article) 

def sent_to_words(sentences):
    yield(gensim.utils.simple_preprocess(str(sentences), deacc=True))  # deacc=True removes punctuations
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

