import boto3
import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import wikipedia

import nlp
import wikidata
import mwapi_queries
import pyperclip


import datetime
import json
from stqdm import stqdm


def benchmark_values(benchmark_data, final_results):
	"""
	Function to benchmark relevant Wikipedia entry values into percentiles based on benchmark data

	Inputs:
	* benchmark_data - dictionary of benchmark images with their page views and page completeness
	* final_results - pd DataFrame of current results of images and candidate Wikipedia entries

	Outputs:
	* pageviewsperc - list of percentile of pageviews for each Wikipedia entry
	* pagecompletenessperc - list of percentile of page completeness for each Wikipedia entry
	"""
	pageviewsperc = []
	pagecompletenessperc = []

	for row in final_results.iterrows():
		if row[1]['page_views'] < np.max(benchmark_data.pageviews.values):
			index = np.where(np.sort(benchmark_data.pageviews.values) > row[1]['page_views'])[0][0]
			perc = np.sort(benchmark_data.page_views_percentile.values)[index]
			pageviewsperc.append(perc)
		else:
			pageviewsperc.append(100)

		if row[1]['page_completeness'] < np.max(benchmark_data.pagecompleteness.values):
			index = np.where(np.sort(benchmark_data.pagecompleteness.values) > row[1]['page_completeness'])[0][0]
			perc = np.sort(benchmark_data.page_completeness_percentile.values)[index]
			pagecompletenessperc.append(perc)
		else:
			pagecompletenessperc.append(100)

	return pageviewsperc, pagecompletenessperc

def calculate_results(images, use_wikimedia, language_process, language_search):
	if use_wikimedia == "Yes":
		use_wikimedia = True
	else:
		use_wikimedia = False

	st.header("Calculating results")
	use_ner = True

	if use_ner:
		ner_filter = True
		# load tagger
		from flair.models import SequenceTagger
		st.write("Loading NER model")
		tagger = SequenceTagger.load("flair/ner-english-fast")
		
	else:

		ner_filter = False
		tagger = None

	st.write("Collecting benchmark data")

	with urllib.request.urlopen("https://measuring-impact-wikimedia.s3.eu-west-1.amazonaws.com/benchmark-data/25-06-2022/benchmark_data.json") as url:
		benchmark_data = json.loads(url.read().decode()) 

	benchmark_data_df, summaries = mwapi_queries.process_benchmark_data(benchmark_data=benchmark_data)
 
	image_corpus = {}
	final_results = pd.DataFrame()

	# JUST FOR DEBUGGING _ DELETE
	images = images[:250]

	first_words = []
	for i in stqdm(range(len(images))):
		image = images[i]
		if use_wikimedia:
			first_words.append(image[5:].split(' ')[0])

		print("***********************************************")
		print("Processing image {}, {} out of {}".format(image, i+1, len(images)))
		final_words = nlp.create_image_main_words(image_title=image,
												  language=language_process,
												  nlp_filter=None,
												  ner_filter = ner_filter,
												  tagger = tagger
												  )
		search_results = mwapi_queries.word_search_query_compound(words=final_words,
																  image_title=image,
																  language_process=language_process,
																  language_search=language_search,
																  use_wikimedia=use_wikimedia)
		print(search_results)
		image_corpus[image] = mwapi_queries.entry_text_query(pages=search_results,
															 language=language_search)

		if len(search_results) > 0:
			tf_idf = nlp.tf_idf(page_summaries=image_corpus[image],
								training_summaries=summaries,
								final_words=final_words,
								image=image)

			tf_idf.loc[:,'page_views'] = np.tile(None, len(tf_idf))
			tf_idf.loc[:,'page_completeness'] = np.tile(None, len(tf_idf))
			tf_idf.loc[:,'language'] = np.tile(None, len(tf_idf))

			views = {}
			for page in search_results:
				page_views = mwapi_queries.page_views_query_requests(page, language=language_search, start_date="20220101")
				tf_idf.loc[tf_idf.entry==page,'page_views'] = int(page_views)
				tf_idf.loc[tf_idf.entry==page,'page_completeness'] = mwapi_queries.page_completeness(page=page)
				similar_images = mwapi_queries.image_uniqueness(page=page,main_image=image,relevant_words=final_words)
				tf_idf.loc[tf_idf.entry==page, 'image_uniqueness'] = np.tile(similar_images, len(tf_idf[tf_idf.entry == page]))
				tf_idf.loc[tf_idf.entry==page, 'language'] = language_search

		else:
			tf_idf = None

		final_results = final_results.append(tf_idf)

	if len(final_results) == 0:
		st.warning("Not able to find relevant results for these settings")

	else:
		# Change absolute values of page views and page completeness to relative percentiles based on benchmark data
		pageviewsperc, pagecompletenessperc = benchmark_values(benchmark_data=benchmark_data_df, 
															   final_results=final_results)

		final_results['page_views_percentile'] = pageviewsperc
		final_results['page_completeness_percentile'] = pagecompletenessperc

		# Combine all components into one final score
		final_results['final_score'] = 0.25*final_results['word_relevance'] \
									 + 0.25*final_results['page_views_percentile']*0.01 \
									 + 0.25*final_results['page_completeness_percentile']*0.01 \
									 + 0.25*final_results['image_uniqueness']
		final_results.final_score = final_results.final_score.round(2)
		final_results = final_results.reset_index(drop=True)

		st.write("Finished calculating results")
		st.dataframe(final_results)
		# Save results to csv
		file_name = "impact_results_{}.csv".format(datetime.datetime.now())
		st.download_button("Download csv file of results", final_results.to_csv(index=False), file_name, "text/csv")
		# st.write("Final results saved to {}".format("final_results_portraits_stlit.csv"))
		# final_results.to_csv('final_results_portraits_stlit.csv',index=False)


st.title("Create Open Access Analysis Data")

use_wikimedia = st.radio("Is the collection uploaded on Wikimedia?", 
							("Yes", "No"))

if use_wikimedia == "Yes":
	collection_id = st.text_input("Input the Wikimedia ID of the collection")
	if len(collection_id) > 0:
		images = wikidata.images_in_collection(collection_wikipedia_id=collection_id, retrieval_limit=500)
		
		language_process = st.radio("Select the main language for the data analysis",
								 ("english", "spanish", "italian", "german"))

		language_search = st.radio("Select the main language of Wikipedia entries to search",
								 ("english", "spanish", "italian", "german"))

		confirm_lang = st.button("Confirm language selection")

		if confirm_lang:
			calculate_results(images, use_wikimedia, language_process, language_search)	


elif use_wikimedia == "No":

	# use_paste = st.radio("Is the collection in an xml file?", 
	# 						("Yes", "No"))

	# if use_paste == "Yes":
	st.subheader("Instructions")
	st.write("You can upload your collection from a file which should contain a unique id per item as well as any fields containing metadata about your items such as title, author or locations")
	st.write("The file should be in xml format")
	st.write("For best results, ensure any non-relevant columns are not present in the file")
	# file_format = st.radio("Select the file format (Note that if the format does not appear in the list below, the format is not yet supported)", 
	# 						(".csv", ".xlsx"))
	file_format = ".xml"
	uploaded_file = st.file_uploader("Choose the file containing your collection")

	if uploaded_file is not None:
		if uploaded_file.name.split(".")[-1] != file_format[1:]:
			st.subheader("Error")
			st.text('Your file type .{} does not match the selected extension .{}'.format(uploaded_file.name.split(".")[-1], file_format[1:]))
		# if file_format == ".csv":
		# 	file_df = pd.read_csv(uploaded_file)
		# elif file_format == ".xlsx":
		# 	file_df = pd.read_excel(uploaded_file, engine='openpyxl', header=1)
		if file_format == ".xml":
			file_df = pd.read_xml(uploaded_file)
		st.dataframe(file_df)

		file_df["aggregate_string"] = file_df.apply(lambda x: str([value for value in x if not pd.isnull(value)]), axis=1)
		images = file_df.aggregate_string.values

		language_process = st.radio("Select the main language for the data analysis",
								 ("english", "spanish", "italian", "german"))

		language_search = st.radio("Select the main language of Wikipedia entries to search",
								 ("english", "spanish", "italian", "german"))

		confirm_lang = st.button("Confirm language selection")

		if confirm_lang:
			calculate_results(images, use_wikimedia, language_process, language_search)	
			

	# elif use_paste == "No":

	# 	st.write("Copy the data you want to use")
	# 	data_ready = st.radio("Have you copied the data you want to use?", 
	# 						("No", "Yes"))
	# 	if data_ready == "Yes":
	# 		from io import StringIO

	# 		data = StringIO(pyperclip.paste())
	# 		df = pd.read_csv(data, sep='delimiter', header=None)
	# 		st.dataframe(df)

