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
import gzip
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

def  word2vector(train,test):
	
	stop_words = stopwords.words('english')
	def preprocess(text):
    		text = text.lower()
    		doc = word_tokenize(text)
    		doc = [word for word in doc if word not in stop_words]
    		doc = [word for word in doc if word.isalpha()]
    		return doc

# Fetch ng20 dataset
	short_train = open(train,"r").read().decode('utf-8')
        short_test =open(test,"r").read().decode('utf-8')
	ng20 =[short_train,short_test]
	# text and ground truth labels

	texts = ng20

	corpus = [preprocess(text) for text in texts]


	sims = {'ng20': {}, 'snippets': {}}
	dictionary = corpora.Dictionary(corpus)
	corpus_gensim = [dictionary.doc2bow(doc) for doc in corpus]
	tfidf = TfidfModel(corpus_gensim)
	corpus_tfidf = tfidf[corpus_gensim]
	lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
	lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
	sims['ng20']['LSI'] = np.array([lsi_index[lsi[corpus_tfidf[i]]]
                                for i in range(len(corpus))])

	def document_vector(word2vec_model, doc):

    		# remove out-of-vocabulary words

    		doc =[word for word in doc if word in word2vec_model.vocab]
    		return np.mean(word2vec_model[doc], axis=0)

	def has_vector_representation(word2vec_model, doc):
    		"""check if at least one word of the document is in the
    		word2vec dictionary"""
    		return not all(word not in word2vec_model.vocab for word in doc)

	sims['ng20']['centroid'] = cosine_similarity(np.array([document_vector(word2vec_model, doc) for doc in corpus]))
	return sims['ng20']['centroid'][0][1]

if __name__ == "__main__":

	glove_input_file = '/home/sys3002/Desktop/glove.6B.100d.txt'
	word2vec_output_file = '/home/sys3002/Desktop/glove.6B.100d.txt.word2vec'
	glove2word2vec(glove_input_file, word2vec_output_file)
	filename = '/home/sys3002/Desktop/glove.6B.100d.txt.word2vec'
        word2vec_model = word2vec.KeyedVectors.load_word2vec_format(filename, binary=False)
        train=raw_input("enter_train textfile path: ")
        test=raw_input("enter _test textfile path: ")
        output= word2vector(train,test)
	print output

