import string
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def create_image_main_words(image_title, nlp_filter=None):
    image_title_words = []
    remove_digits = str.maketrans('', '', string.digits)
    remove_punctuation = str.maketrans('','',string.punctuation)

    if image_title[-4:] == 'jpeg':
        image_words = image_title[5:-4].translate(remove_digits)
    else:
        image_words = image_title[5:-3].translate(remove_digits)

    image_words = image_words.translate(remove_punctuation)
    image_title_words = image_words.split()
    image_title_words = [word.lower() for word in image_title_words]

    if nlp_filter:
    	final_words = apply_nlp_filter(words=image_title_words, 
    								   nlp_filter=nlp_filter)

    stopwords_eng = set(stopwords.words('english'))

    final_words = [word for word in final_words if not word in stopwords_eng]
    final_words = set(final_words)
    
    return final_words

def preprocess_summaries(page_summaries):
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
	
	vectorizer = TfidfVectorizer()
	vectors = vectorizer.fit_transform(all_summaries)
	feature_names = vectorizer.get_feature_names()
	dense = vectors.todense()
	denselist = dense.tolist()

	df = pd.DataFrame(denselist, columns=feature_names)

	relevant_words = [word for word in final_words if word in df.columns.values]
	relevant_df = df[relevant_words]
	relevant_df['word_relevance'] = relevant_df.sum(axis=1).values

	relevant_df.loc[:,'image'] = image

	ordered_indexes = list(relevant_df.sort_values(by='word_relevance', ascending=False).index.values)

	print("RESULTS FOR IMAGE {}:".format(image))
	print("Relevant words are {}".format(relevant_words))
	ordered_pages = []
	# for i, page in enumerate(all_pages):
	# 	import pdb
	# 	pdb.set_trace()
	# 	relevance = relevant_df['word_relevance'].iloc[ordered_indexes[i]]
	# 	if relevance > 0:
	# 		print("Page {} has a relevance of {}".format(all_pages[ordered_indexes[i]],relevance))
	# 	ordered_pages.append(all_pages[ordered_indexes[i]])
	# 	print("Appended page", all_pages[ordered_indexes[i]], "in order", i, "with relevance", relevance)

	relevant_df.loc[:, 'entry'] = all_pages

	relevant_df['relevant_words'] = [relevant_words for i in relevant_df.index]
	relevant_df = relevant_df[relevant_df['word_relevance'] > 0]
	relevant_df = relevant_df.sort_values(by='word_relevance', ascending=False)
	relevant_df = relevant_df[['image','entry','word_relevance', 'relevant_words']]
	relevant_df = relevant_df[relevant_df.entry.isin(pages)]

	return relevant_df


def apply_nlp_filter(words, nlp_filter):
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

