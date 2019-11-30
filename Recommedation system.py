#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget -nc -nv https://ucsb.box.com/shared/static/c5ulwkcaka7hych2ou8bjwz7p8gcvxyw.json -O data/business.json')


# In[2]:


get_ipython().system(' head -n1 data/business.json')


# In[3]:


import pandas as pd
import json
import numpy as np
get_ipython().system(' wget -nc -nv https://ucsb.box.com/shared/static/ne0i3no3ep9z8rjtob9ywm4uhtfrmeyn.pkl -O train_review.pkl')

import pickle  
train_review = pickle.load(open('train_review.pkl', "rb" ))


# In[4]:


review=pd.DataFrame(train_review)
review=review.sort_values(['business_id'])
newreview=review.groupby(['business_id']).head(20) ##need to be resample


# In[5]:


newreview


# In[6]:


review.groupby(['business_id']).mean() # see the mean rate of these 94 resturants 


# In[7]:


newtext=[]
for i in range(len(newreview)):
    newtext.append(newreview.iloc[i,7])


# In[8]:


get_ipython().system(' pip install nltk')
# download dictionaty of stop words
import nltk
nltk.download('stopwords')
nltk.download('punkt') # tockenizer


# In[9]:


import re
import string
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()


# ##  Tokenizing Text
# 
# 
# Sentence tokenization is the process of splitting a text corpus into sentences that act as the first level of tokens which the corpus is comprised of. This is also known as sentence segmentation, because we try to segment the text into meaningful sentences. Any text corpus is a body of text where each paragraph comprises several sentences.
# 
# We will use the `nltk` framework, which provides various interfaces for performing sentence tokenization.

# In[10]:


def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens


# ## Expand contractions
# 
# Contractions are shortened version of words or syllables. They exist in either written or spoken forms. Shortened versions of existing words are created by removing specific letters and sounds. In case of English contractions, they are often created by removing one of the vowels from the word. 
# 
# A vocabulary for contractions and their corresponding expanded forms that you can access in the file `contractions.py` in a Python dictionary (which we again download from the textbook repo).
# 
# Contraction map is a dictionary:

# In[11]:


get_ipython().system(' wget -nc -nv https://raw.githubusercontent.com/dipanjanS/text-analytics-with-python/master/Old-First-Edition/source_code/Ch04_Text_Classification/contractions.py')
from contractions import CONTRACTION_MAP
CONTRACTION_MAP


# In[13]:


def expand_contractions(text, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)                                if contraction_mapping.get(match)                                else contraction_mapping.get(match.lower())
        # not sure why below is there
        # expanded_contraction = first_char+expanded_contraction[1:] 
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# In[14]:


newtext = [expand_contractions(sentence, CONTRACTION_MAP) for sentence in newtext]
newtext


# ## Removing Special Characters
# 
# One important task in text normalization involves removing unnecessary and special characters. These may be special symbols or even punctuation that occurs in sentences. This step is often performed before or after tokenization. The main reason for doing so is because often punctuation or special characters do not have much significance when we analyze the text and utilize it for extracting features or information based on NLP and ML.

# In[15]:


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# In[16]:


newtext = [remove_special_characters(sentence) for sentence in newtext]
newtext


# ## Removing Stopwords
# Stopwords are words that have little or no significance. They are usually removed from text during processing so as to retain words having maximum significance and context. Stopwords are usually words that end up occurring the most if you aggregated any corpus of text based on singular tokens and checked their frequencies.

# In[17]:


from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


# In[18]:


ENGLISH_STOP_WORDS


# In[19]:


def remove_stopwords(text,stopword_list):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


# In[20]:


newtext = [remove_stopwords(sentence,
                         stopword_list=list(ENGLISH_STOP_WORDS))\
        for sentence in newtext]


# In[21]:


nltk.download(['averaged_perceptron_tagger',
               'universal_tagset',
               'wordnet'])


# In[22]:


from nltk import pos_tag
from nltk.corpus import wordnet as wn

# Annotate text tokens with POS tags
def pos_tag_text(text):
    
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    tagged_text = pos_tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text


# In[23]:


# lemmatize text based on POS tags    
def lemmatize_text(text):
    text = tokenize_text(text)
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word                     
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text


# In[24]:


# Text normalization pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import re

def keep_text_characters(text):
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def normalize_text(text,tokenize=False):
    text = expand_contractions(text, CONTRACTION_MAP)
    text = lemmatize_text(text)
    text = remove_special_characters(text)
    text = text.lower()
    text = remove_stopwords(text,ENGLISH_STOP_WORDS)
    text = keep_text_characters(text)

    return text


# In[25]:


newreview_corpus_norm = [normalize_text(text) for text in newtext]


# In[26]:


testreview_corpus_norm=newreview_corpus_norm
import math
testreview_corpus_norm
for i in range(0,math.floor(len(testreview_corpus_norm)/20),1):
    testreview_corpus_norm[i : i+20] = [''.join(testreview_corpus_norm[i : i+20])] 
testreview_corpus_norm


# In[27]:


testreview_corpus_norm[93]#check


# In[28]:


# TF-IDF 
import numpy as np
from feature_extractors import tfidf_transformer
from feature_extractors import bow_extractor    

def tf_idf(corpus):
    # Bag of words construction
    bow_vectorizer, bow_features = bow_extractor(corpus=corpus)
    # feature names
    feature_names = bow_vectorizer.get_feature_names()
    # TF-IDF    
    tfidf_trans, tdidf_features = tfidf_transformer(bow_features)
    tdidf_features = np.round(tdidf_features.todense(),2)
    return((tdidf_features, feature_names))


# In[29]:


tdidf_features,feature_names = tf_idf(testreview_corpus_norm) 


# # Topic Modeling
# 
# *Topic models* have been designed specifically for the purpose of extracting various distinguishing concepts or topics from a large corpus containing various types of documents.
# 
# Topic modeling is a *unsupervised* learning technique since involves extracting features from document terms to generate clusters or groups of terms that are distinguishable from each other, and these cluster of words form topics or concepts. 

# In[30]:


X = tdidf_features.T
X.shape # (words, documents)


# In[31]:


X


# In[32]:


## Non-negative Matrix Factorization
def non_negative_marix_decomp(n_components,train_data):
    import sklearn.decomposition as skld
    model = skld.NMF(n_components=n_components, 
                     init='nndsvda', max_iter=500, 
                     random_state=0)
    W = model.fit_transform(train_data)
    H = model.components_
    nmf = (W,H)
    return(nmf)


# In[33]:


r = 5 # no. of topics
W_topic5,H_topic5 =     non_negative_marix_decomp(n_components = r, train_data = X) 


# In[34]:


r = 10 # no. of topics
W_topic10,H_topic10 =     non_negative_marix_decomp(n_components = r, train_data = X) 

H_topic10 /= H_topic10.sum(0)


# In[35]:


get_ipython().system(' wget -nc -nv https://raw.githubusercontent.com/dipanjanS/text-analytics-with-python/ed5ea8068428fec37d1d06ec40cb9d64c6336d77/Old-First-Edition/source_code/Ch05_Text_Summarization/topic_modeling.py')


# In[36]:


def extract_topic_top_words(W, all_words, num_top_words=10):
    
    num_words, num_topics = W.shape
    
    assert num_words == len(all_words)
    
    for t in range(0, num_topics):
        top_words_idx = np.argsort(W[:,t])[::-1]  # descending order
        top_words_idx = top_words_idx[:num_top_words]
        top_words = [all_words[k] for k in top_words_idx]
        top_words_shares = W[top_words_idx, t]
        print('# Topic', t+1)
        for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
            print(word, share)
        print('\n')
        
extract_topic_top_words(W_topic5, feature_names)


# In[37]:


extract_topic_top_words(W_topic10, feature_names)


# In[38]:


num_topics, num_reviews = H_topic10.shape

H10df = pd.DataFrame(H_topic10, 
                     index=['topic'+str(one) for one in range(1, num_topics+1)], 
                     columns=['review'+str(one) for one in range(1, num_reviews+1)])


# In[39]:


H_topic10


# In[40]:


H10df


# In[41]:


review_corr = H10df.corr()

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(18,18))
ax = sns.heatmap(review_corr, square=True, cmap="coolwarm");

