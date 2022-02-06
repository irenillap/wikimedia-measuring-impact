import streamlit as st
import pandas as pd
import numpy as np
import wikipedia

import os

import main

st.title('Collection Images Suggestions')

sidebar_text = st.sidebar.text(
    'Displaying results of model'
)


weight1 = st.sidebar.slider(
	'What weight would you like to place on word relevance?',
	0.00,1.00,0.01
)

weight2 = st.sidebar.slider(
	'What weight would you like to place on entry views?',
	0.00,1.00,0.01
)

weight3 = st.sidebar.slider(
	'What weight would you like to place on entry completeness?',
	0.00,1.00,0.01
)

weight4 = st.sidebar.slider(
	'What weight would you like to place on image uniqueness?',
	0.00,1.00,0.01
)

# eq_weighting = st.sidebar.button('Click here for equal weighting')
# if eq_weighting:
# 	weight1 = 0.25
# 	weight2 = 0.25
# 	weight3 = 0.25
# 	weight4 = 0.25


confirmation_button = st.sidebar.button('Confirm weights and rerun')

# suggested_results = pd.read_csv('final_results_25072021.csv')
# suggested_results = pd.read_csv('results_final_pilot_coll_compound_250721.csv')
# suggested_results = pd.read_csv('final_results_11082021.csv')
suggested_results = pd.read_csv('final_results.csv')

if confirmation_button:
	suggested_results['final_score'] = weight1*suggested_results['word_relevance'] \
									 + weight2*suggested_results['page_views_percentile']*0.01 \
									 + weight3*suggested_results['page_completeness_percentile']*0.01 \
									 + weight4*suggested_results['image_uniqueness']

st.write('Current weights:', weight1, weight2, weight3, weight4)
selected_indices = st.multiselect('Select images:', suggested_results.image.unique())
selected_rows = pd.DataFrame(suggested_results[suggested_results.image.isin(selected_indices)])
if len(selected_indices)>0:
	st.dataframe(data=selected_rows, width=1024, height=768)
	slider_max = selected_rows.shape[0]
else:
	selected_rows = suggested_results
	slider_max = suggested_results.shape[0]

if len(selected_rows) > 10:
	slider_num = st.slider("Select number of top suggestions from dataset", 1, min(slider_max,100))
else:
	slider_num = 0

selected_rows = selected_rows.sort_values(by='final_score', ascending=False)
selected_rows = selected_rows.reset_index(drop=True)

for i, result in selected_rows.head(max(10,slider_num)).iterrows():
	st.subheader('SUGGESTION {}'.format(i+1))
	st.write('Image', result['image'], 'could be placed in Wikipedia entry', result['entry'], 'with a total score of', round(result['final_score'],2))
	st.write('It shows a word relevancy of', round(result['word_relevance'],2), 'for the following words:', result['relevant_words'])
	st.write('The page has had an average of', result['page_views'], 'monthly views since the start of 2020 which is higher than', result['page_views_percentile'], 'percent of other entries.' )
	st.write('It currently has approximately', result['page_completeness'], 'words which is higher than', result['page_completeness_percentile'], 'percent of other entries.')
	st.write('It has an image uniqueness of', result['image_uniqueness'])
	with st.beta_expander("See Wikipedia summary of entry"):
	    st.write(wikipedia.summary(result['entry']))
	