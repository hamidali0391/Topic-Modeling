import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)
import nltk
nltk.download('wordnet')
import pandas as pd


data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
# We only need the Headlines text column from the data
data_text = data[:300000][['headline_text']];
data_text['index'] = data_text.index

documents = data_text

processed_docs=np.load("processed_docs.npy",allow_pickle=True)

'''
Create a dictionary from 'processed_docs' containing the number of times a word appears 
in the training set using gensim.corpora.Dictionary and call it 'dictionary'
'''
dictionary = gensim.corpora.Dictionary(processed_docs)

# apply dictionary.filter_extremes() with the parameters mentioned above
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)

'''
Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
words and how many times those words appear. Save this to 'bow_corpus'
'''
bow_corpus=[dictionary.doc2bow(doc) for doc in processed_docs]

'''
Create tf-idf model object using models.TfidfModel on 'bow_corpus' and save it to 'tfidf'
'''
from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)

'''
Apply transformation to the entire corpus
'''
corpus_tfidf = tfidf[bow_corpus]


lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                    num_topics = 10, 
                                    id2word = dictionary,                                    
                                    passes = 5)


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, 
                                    num_topics = 10, 
                                    id2word = dictionary,                                    
                                    passes = 5)
document_num = 4310

# Our test document is document number 4310



for index, score in sorted(lda_model[bow_corpus[document_num]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))


for index, score in sorted(lda_model_tfidf[bow_corpus[document_num]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))