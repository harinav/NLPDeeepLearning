from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
import numpy as np
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import gensim.models.keyedvectors as word2vec
from gensim.models import Word2Vec
filename = '/home/sys3002/Desktop/GoogleNews-vectors-negative300.bin.gz'
word2vec_model=word2vec.KeyedVectors.load_word2vec_format(filename, binary=True)
download('punkt')
download('stopwords')

stop_words = stopwords.words('english')

def preprocess(text):
   
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()]
    return doc

# Fetch ng20 dataset
ng20 = fetch_20newsgroups(subset='train',
# text and ground truth labels
                          remove=('headers', 'footers', 'quotes'))
print ng20
texts, y = ng20.data, ng20.target

corpus = [preprocess(text) for text in texts]
def filter_docs(corpus, texts, labels, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels)

corpus, texts, y = filter_docs(corpus, texts, y, lambda doc: (len(doc) != 0))

sims = {'ng20': {}, 'snippets': {}}
dictionary = corpora.Dictionary(corpus)
corpus_gensim = [dictionary.doc2bow(doc) for doc in corpus]
tfidf = TfidfModel(corpus_gensim)
corpus_tfidf = tfidf[corpus_gensim]
lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
sims['ng20']['LSI'] = np.array([lsi_index[lsi[corpus_tfidf[i]]]
                                for i in range(len(corpus))])

word2vec_model.init_sims(replace=True)

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    print doc
    return np.mean(word2vec_model[doc], axis=0)



def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)

corpus, texts, y = filter_docs(corpus, texts, y,
                               lambda doc: has_vector_representation(word2vec_model, doc))

sims['ng20']['centroid'] = cosine_similarity(np.array([document_vector(word2vec_model, doc)]))
