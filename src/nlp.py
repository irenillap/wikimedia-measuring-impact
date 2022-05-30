import string
import numpy as np
import pandas as pd
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from flair.data import Sentence
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import jieba
import nagisa

def create_image_main_words(image_title, stopword, nlp_filter=None, ner_filter = None, tagger = None, output_type = ['PER','LOC','ORG','MISC']):
        """

	Function to extract key words from the relevant image title. To do this, remove all digits,
	punctuation and stopwords from image title. Also uses an nlp filter if there is one to place
	more emphasis on words relevant to image collection (eg 'castle' for landscape images)
	
	Inputs:
	* image_title - string of image title as appears on Wikimedia
	* nlp_filter - string to call on manually created nlp filter if needed
	* ner_filter - whether to use a ner filter. if use, must specify a tagger
	* tagger - a ner model
	* output_type - type of named entity to keep. possible values are ['PER','LOC','ORG','MISC'], where PER = person name; LOC = location name; ORG = organization name; MISC = other name. default to ['PER','LOC','ORG','MISC']
    
	Outputs:
	* final_words - set of final relevant words extracted from title
	
	"""
        image_title_words = []
        remove_digits = str.maketrans('', '', string.digits)
        remove_punctuation = str.maketrans('','',string.punctuation)

        if image_title[-4:] == 'jpeg':
                image_words = image_title[5:-4].translate(remove_digits)
        else:
                image_words = image_title[5:-3].translate(remove_digits)
        
        if ner_filter:
            if tagger == None:
                raise Exception("must specify a tagger to use ner filter")
            image_title_words = apply_ner_filter(image_words, tagger = tagger, output_type = output_type)
        
        image_words = image_words.translate(remove_punctuation)
        image_title_words = image_words.split()
        image_title_words = [word.lower() for word in image_title_words]

        if nlp_filter:
                final_words = apply_nlp_filter(words=image_title_words, 
                                                           nlp_filter=nlp_filter)

        stopwords = set(stopword)

        final_words = [word for word in final_words if not word in stopwords]
        final_words = set(final_words)

        return final_words

def preprocess_summaries(page_summaries):
	"""
	Function to create full corpus of words from candidate pages summaries

	Inputs:
	* page_summaries - dictionary of image being studied and summary text of all 
	candidate Wikipedia entries

	Outputs:
	* all_summaries - list of all words in all relevant summaries
	"""
	all_summaries = []

	for image, summary in page_summaries.items():
		remove_digits = str.maketrans('', '', string.digits)
		remove_punctuation = str.maketrans('','',string.punctuation)

		new_summary = summary.translate(remove_digits)
		new_summary = new_summary.translate(remove_punctuation)

		summary_words = new_summary.split()
		summary_words = [word.lower() for word in summary_words]
		
		all_summaries.append(summary_words)

	return all_summaries


def tf_idf(page_summaries, training_summaries, final_words, image):
	"""
	Function to calculate tf-idf score defining word relevance for image and candidate Wikipedia entry

	Inputs:
	* page_summaries - dictionary of image being studied and summary text of all 
	candidate Wikipedia entries
	* training_summaries - dictionary of all benchmark summaries
	* final_words - set of relevant words extracted from image
	* image - string of image title as appears on Wikimedia

	Outputs:
	* relevant_df - pd DataFrame of numerical relevance of relevant words
	"""
	all_summaries = []
	all_pages = []
	pages = []

	all_corpus = page_summaries.copy()   # start with keys and values of x
	all_corpus.update(training_summaries)

	for page, summary in all_corpus.items():
		all_summaries.append(summary)
		all_pages.append(page)

	for page, summary in page_summaries.items():
		pages.append(page)
	
	#vectorizer = TfidfVectorizer()
	vectorizer = TfidfVectorizer(tokenizer = identity_tokenizer,lowercase=False)
	vectors = vectorizer.fit_transform(all_summaries)
	feature_names = vectorizer.get_feature_names()
	#dense = vectors.todense()
	#denselist = dense.tolist()

	#df = pd.DataFrame(denselist, columns=feature_names)

	relevant_words = [word for word in final_words if word in feature_names]
	#relevant_df = df[relevant_words].copy()
	relevant_df = pd.DataFrame(vectors[:,[vectorizer.vocabulary_[word] for word in final_words if word in feature_names]].sum(axis=1),columns = ['word_relevance'])

	relevant_df.loc[:,'image'] = image

	ordered_indexes = list(relevant_df.sort_values(by='word_relevance', ascending=False).index.values)

	print("RESULTS FOR IMAGE {}:".format(image))
	print("Relevant words are {}".format(relevant_words))
	ordered_pages = []

	relevant_df.loc[:, 'entry'] = all_pages

	relevant_df.loc[:,'relevant_words'] = pd.Series([relevant_words for i in relevant_df.index],dtype = 'object')
	relevant_df = relevant_df[relevant_df['word_relevance'] > 0]
	relevant_df = relevant_df.sort_values(by='word_relevance', ascending=False)
	#relevant_df = relevant_df[['image','entry','word_relevance', 'relevant_words']]
	relevant_df = relevant_df[relevant_df.entry.isin(pages)]

	return relevant_df


def apply_nlp_filter(words, nlp_filter):
	"""
	Function to apply nlp filter if required. Applying the filter creates compound relevant words 
	e.g. if 'Harlem Castle from East' is title, normal words would pick up ['Harlem', 'Castle', 'East']. This 
	function will also append ['Harlem Castle'] to the list.

	Inputs:
	* words - set of relevant words extracted from an image title 
	* nlp_filter - string of applied nlp filter

	Outputs:
	* new_words - updated set of relevant words for image title
	"""
	if nlp_filter == 'landscape':
		keywords = ['castle', 'river', 'abbey', 'hill', 'shire', 'baths', 'bridge', 'church', 'waterfall','mountain', 'vale']

	new_words = words.copy()
	for i, word in enumerate(words):
		if word in keywords:
			if len(words) > i+1 and words[i+1] == 'of':
				new_word = word + " " + words[i+1] + " " + words[i+2]
			else:
				new_word = words[i-1] + " " + word
			new_words.append(new_word)
			print("Appending {} to words".format(new_word))

	return new_words
	
def apply_ner_filter(words, tagger, output_type):
  
	"""
	Function to apply a named entity recognition filter to get rid of irrelevant words

	Input:
	* words (image_title) - string of image title as appears on Wikimedia
	* tagger - a ner model
	* output_type - type of named entity to keep. possible values are ['PER','LOC','ORG','MISC'], where PER = person name; LOC = location name; ORG = organization name; MISC = other name

	Output:
	* filtered_words - updated set of named entity words for image title

	example:
	ner_filter('George Washington went to Washington',['PER'],SequenceTagger.load("flair/ner-english-fast"))
	-> 'George Washington'
	"""
	return ner_tokenization(words, tagger, output_type, return_nonnerwords = False)

def identity_tokenizer(text):
	"""
	helper function for sklearn tfidfvectorizer
	"""
	return text

def base_tokenization(text, stopwords = None):

    doc = re.findall(r'(?u)\b\w\w+\b',text)
    
    if stopwords != None:
        return [word for word in doc if not word in stopwords]
    else:
        return doc

def chinese_tokenization(text, stopwords = None):

    doc = jieba.lcut(text)

    if stopwords != None:
        return [word for word in doc if not word in stopwords]
    else:
        return doc

def japanese_tokenization(text, stopwords = None):

    doc = nagisa.tagging(text)

    if stopwords != None:
        return [word for word in doc.words if not word in stopwords]
    else:
        return doc.words

def ner_tokenization(words, tokenization, tagger, output_type, return_nonnerwords = True):
	  
	"""
	Function to apply a named entity recognition tokenization

	Input:
	* words (image_title) - string of image title as appears on Wikimedia
	* tagger - a ner model
	* output_type - type of named entity to keep. possible values are ['PER','LOC','ORG','MISC'], where PER = person name; LOC = location name; ORG = organization name; MISC = other name
	* return_nonnerwords - whether to return non-ner words

	Output:
	* tokenized_words
	"""
	sentence = Sentence(words)
	
	tagger.predict(sentence.to(device))
	
	nerwords = [entity.text for entity in sentence.get_spans('ner') if entity.tag in output_type]
	
	if return_nonnerwords:

		idx_list = [s.idx for span in sentence.get_spans('ner') for s in span]
		
		processed_sentence = ' '.join([token.text for token in sentence.tokens if token.idx not in idx_list])

		non_nerwords = tokenization(processed_sentence)

		return non_nerwords+nerwords
	
	else:
		return nerwords

