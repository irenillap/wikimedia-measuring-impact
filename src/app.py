import streamlit as st
import pandas as pd
import numpy as np
import wikipedia
import matplotlib.pyplot as plt

import os

import main

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def read_in_csv(suggested_results_file):
	suggested_results = pd.read_csv(suggested_results_file)
	st.write('Successfully loaded the input file!')

	return suggested_results

st.title('Collection Images Suggestions')

sidebar_text = st.sidebar.text(
    'Displaying results of model'
)

mode = st.sidebar.radio("What mode do you want?", 
						('Engagement', 'Relevance', 'Wiki enhancement', 'Custom'))

if mode == 'Engagement':
	weight1 = 0.4
	weight2 = 0.3
	weight3 = 0.2
	weight4 = 0.1

elif mode == 'Relevance':
	weight1 = 0.6
	weight2 = 0.2
	weight3 = 0.1
	weight4 = 0.1

elif mode == 'Wiki enhancement':
	weight1 = 0.4
	weight2 = 0.1
	weight3 = 0.1
	weight4 = 0.4

elif mode == 'Custom':

	weight1 = st.sidebar.number_input('Insert a weight for word relevance', value=0.4, min_value=0.0, max_value=1.0)
	weight2 = st.sidebar.number_input('Insert a weight for entry views', value=0.3, min_value=0.0, max_value=1.0)
	weight3 = st.sidebar.number_input('Insert a weight for entry completeness', value=0.2, min_value=0.0, max_value=1.0)
	weight4 = st.sidebar.number_input('Insert a weight for image uniqueness', value=0.1, min_value=0.0, max_value=1.0)

confirmation_button = st.sidebar.button('Confirm custom weights and rerun')

suggested_results_file = st.file_uploader("Choose the file containing your results")
suggested_results = None
if suggested_results_file is not None:
	suggested_results = read_in_csv(suggested_results_file)

if suggested_results is not None:
	if confirmation_button:
		suggested_results['final_score'] = weight1*suggested_results['word_relevance'] \
										 + weight2*suggested_results['page_views_percentile']*0.01 \
										 + weight3*suggested_results['page_completeness_percentile']*0.01 \
										 + weight4*suggested_results['image_uniqueness']

	st.write('-------------------------------------')
	st.write('Current weights:')
	st.write('Word relevance weight', round(weight1,2))
	st.write('Entry views weight', round(weight2,2))
	st.write('Entry completeness weight', round(weight3,2))
	st.write('Image uniqueness weight', round(weight4,2))

	mode = st.radio("Select mode:",
	     ('Aggregate Data', 'Explore Individual Images', 'Effort Analysis')
		)

	if mode == 'Explore Individual Images':
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

	elif mode == 'Aggregate Data':
		assumed_entry_accuracy = st.number_input('Insert an assumed entry accuracy', value=0.6, min_value=0.0, max_value=1.0)
		assumed_top_entries_actioned = st.number_input('Insert the assumed number of top entries actioned', value=int(len(suggested_results.index.values) * assumed_entry_accuracy), 
														min_value=0, max_value=int(len(suggested_results.index.values) * assumed_entry_accuracy))

		impact_confirmation_button = st.button('Confirm input parameters')

		if impact_confirmation_button:
			suggested_results['final_score'] = weight1*suggested_results['word_relevance'] \
										 + weight2*suggested_results['page_views_percentile']*0.01 \
										 + weight3*suggested_results['page_completeness_percentile']*0.01 \
										 + weight4*suggested_results['image_uniqueness']

			suggested_results = suggested_results.sort_values(by='final_score', ascending=False)

			top_entry_number = int(assumed_top_entries_actioned/assumed_entry_accuracy)
			considered_results = suggested_results[:top_entry_number]

			all_tot_views = []
			all_av_views = []
			all_av_relevancy = []

			for i in range(1000):
				random_sampled_results = considered_results.sample(n=int(assumed_top_entries_actioned))

				tot_views = np.sum(random_sampled_results.page_views.values)
				av_views = np.mean(random_sampled_results.page_views.values)
				av_relevancy = np.mean(random_sampled_results.word_relevance.values)

				all_tot_views.append(tot_views)
				all_av_views.append(av_views)
				all_av_relevancy.append(av_relevancy)

			st.header('Aggregate results')

			st.subheader('Estimated total monthly views for all actioned images:')
			st.text("{:,} +/- {:,}".format(round(np.mean(all_tot_views)), round((np.max(all_tot_views) - np.min(all_tot_views))/2)))

			with st.expander("See distribution based on 1000 simulations"):
				fig, ax = plt.subplots()
				ax.hist(all_tot_views, bins=10)
				ax.axvline(np.mean(all_tot_views), color='r')

				st.pyplot(fig)

			st.subheader('Estimated average montly views per image:')
			st.text("{:,} +/- {:,}".format(round(np.mean(all_av_views)), round((np.max(all_av_views) - np.min(all_av_views))/2)))

			with st.expander("See distribution based on 1000 simulations"):
				fig, ax = plt.subplots()
				ax.hist(all_av_views, bins=10)
				ax.axvline(np.mean(all_av_views), color='r')

				st.pyplot(fig)

			st.subheader('Estimated average relevancy per image:')
			st.text("{} +/- {}".format(round(np.mean(all_av_relevancy),2), round(((np.max(all_av_relevancy) - np.min(all_av_relevancy))/2),2)))

			with st.expander("See distribution based on 1000 simulations"):
				fig, ax = plt.subplots()
				ax.hist(all_av_relevancy, bins=10)
				ax.axvline(np.mean(all_av_relevancy), color='r')

				st.pyplot(fig)

	elif mode == 'Effort Analysis':
		assumed_entry_accuracy_effort = st.slider('Insert a range assumed entry accuracy', min_value=0.0, max_value=1.0, value=(0.5,0.7), step=0.05)

		impact_confirmation_button = st.button('Confirm input parameters')

		if impact_confirmation_button:
			suggested_results['final_score'] = weight1*suggested_results['word_relevance'] \
											 + weight2*suggested_results['page_views_percentile']*0.01 \
											 + weight3*suggested_results['page_completeness_percentile']*0.01 \
											 + weight4*suggested_results['image_uniqueness']

			suggested_results = suggested_results.sort_values(by='final_score', ascending=False)
			
			#First calculate for minimum entry accuracy
			max_assumed_correct_results_min = int(assumed_entry_accuracy_effort[0] *  len(suggested_results.index.values))

			simulation_tot_views = {}
			simulation_av_views = {}
			simulation_av_relevancy = {}
			simulation_tot_relevancy = {}

			for actioned_entries in range (1, max_assumed_correct_results_min, 10):
				top_entry_number = int(actioned_entries/assumed_entry_accuracy_effort[0])
				considered_results = suggested_results[:top_entry_number]

				all_tot_views = []
				all_av_views = []
				all_av_relevancy = []
				all_tot_relevancy = []

				for i in range(1000):
					random_sampled_results = considered_results.sample(n=int(actioned_entries))

					tot_views = np.sum(random_sampled_results.page_views.values)
					av_views = np.mean(random_sampled_results.page_views.values)
					av_relevancy = np.mean(random_sampled_results.word_relevance.values)
					tot_relevancy = np.sum(random_sampled_results.word_relevance.values > 0.5)

					all_tot_views.append(tot_views)
					all_av_views.append(av_views)
					all_av_relevancy.append(av_relevancy)
					all_tot_relevancy.append(tot_relevancy)

				simulation_tot_views[actioned_entries] = all_tot_views
				simulation_av_views[actioned_entries] = all_av_views
				simulation_av_relevancy[actioned_entries] = all_av_relevancy
				simulation_tot_relevancy[actioned_entries] = all_tot_relevancy

			#Now calculate for maximum entry accuracy
			max_assumed_correct_results_max = int(assumed_entry_accuracy_effort[1] *  len(suggested_results.index.values))

			simulation_tot_views_max = {}
			simulation_av_views_max = {}
			simulation_av_relevancy_max = {}
			simulation_tot_relevancy_max = {}

			for actioned_entries in range (1, max_assumed_correct_results_max, 10):
				top_entry_number = int(actioned_entries/assumed_entry_accuracy_effort[1])
				considered_results = suggested_results[:top_entry_number]

				all_tot_views = []
				all_av_views = []
				all_av_relevancy = []
				all_tot_relevancy = []

				for i in range(1000):
					random_sampled_results = considered_results.sample(n=int(actioned_entries))

					tot_views = np.sum(random_sampled_results.page_views.values)
					av_views = np.mean(random_sampled_results.page_views.values)
					av_relevancy = np.mean(random_sampled_results.word_relevance.values)
					tot_relevancy = np.sum(random_sampled_results.word_relevance.values > 0.5)

					all_tot_views.append(tot_views)
					all_av_views.append(av_views)
					all_av_relevancy.append(av_relevancy)
					all_tot_relevancy.append(tot_relevancy)

				simulation_tot_views_max[actioned_entries] = all_tot_views
				simulation_av_views_max[actioned_entries] = all_av_views
				simulation_av_relevancy_max[actioned_entries] = all_av_relevancy
				simulation_tot_relevancy_max[actioned_entries] = all_tot_relevancy

			st.subheader('Total estimated views vs entries actioned')
			fig, ax = plt.subplots()
			tot_views_av_array = [np.mean(values) for values in simulation_tot_views.values()]
			tot_views_error_array = [np.std(values) for values in simulation_tot_views.values()]
			tot_views_av_array_max = [np.mean(values) for values in simulation_tot_views_max.values()]
			tot_views_error_array_max = [np.std(values) for values in simulation_tot_views_max.values()]
			ax.scatter(simulation_tot_views.keys(), tot_views_av_array)
			ax.plot(simulation_tot_views.keys(), tot_views_av_array, alpha=0.5)
			ax.errorbar(simulation_tot_views.keys(), tot_views_av_array, yerr=tot_views_error_array, fmt="o",label="Entry accuracy: {}".format(assumed_entry_accuracy_effort[0]))
			ax.scatter(simulation_tot_views_max.keys(), tot_views_av_array_max)
			ax.plot(simulation_tot_views_max.keys(), tot_views_av_array_max, alpha=0.5)
			ax.errorbar(simulation_tot_views_max.keys(), tot_views_av_array_max, yerr=tot_views_error_array_max, fmt="o", label="Entry accuracy: {}".format(assumed_entry_accuracy_effort[1]))
			plt.legend()
			plt.xlabel('Number of actioned entries')
			plt.ylabel('Cumulative estimated total views')

			st.pyplot(fig)

			st.subheader('Estimated average views vs entries actioned')
			fig, ax = plt.subplots()
			av_views_av_array = [np.mean(values) for values in simulation_av_views.values()]
			av_views_error_array = [np.std(values) for values in simulation_av_views.values()]
			av_views_av_array_max = [np.mean(values) for values in simulation_av_views_max.values()]
			av_views_error_array_max = [np.std(values) for values in simulation_av_views_max.values()]
			ax.scatter(simulation_av_views.keys(), av_views_av_array)
			ax.plot(simulation_av_views.keys(), av_views_av_array, alpha=0.5)
			ax.errorbar(simulation_av_views.keys(), av_views_av_array, yerr=av_views_error_array, fmt="o", label="Entry accuracy: {}".format(assumed_entry_accuracy_effort[0]))
			ax.scatter(simulation_av_views_max.keys(), av_views_av_array_max)
			ax.plot(simulation_av_views_max.keys(), av_views_av_array_max, alpha=0.5)
			ax.errorbar(simulation_av_views_max.keys(), av_views_av_array_max, yerr=av_views_error_array_max, fmt="o", label="Entry accuracy: {}".format(assumed_entry_accuracy_effort[1]))
			plt.legend()
			plt.xlabel('Number of actioned entries')
			plt.ylabel('Estimated average views')

			st.pyplot(fig)

			st.subheader('Estimated total entries with > 0.5 relevance vs entries actioned')
			fig, ax = plt.subplots()
			tot_rel_av_array = [np.mean(values) for values in simulation_tot_relevancy.values()]
			tot_rel_error_array = [np.std(values) for values in simulation_tot_relevancy.values()]
			tot_rel_av_array_max = [np.mean(values) for values in simulation_tot_relevancy_max.values()]
			tot_rel_error_array_max = [np.std(values) for values in simulation_tot_relevancy_max.values()]
			ax.scatter(simulation_tot_relevancy.keys(), tot_rel_av_array)
			ax.plot(simulation_tot_relevancy.keys(), tot_rel_av_array, alpha=0.5)
			ax.errorbar(simulation_tot_relevancy.keys(), tot_rel_av_array, yerr=tot_rel_error_array, fmt="o", label="Entry accuracy: {}".format(assumed_entry_accuracy_effort[0]))
			ax.scatter(simulation_av_relevancy_max.keys(), tot_rel_av_array_max)
			ax.plot(simulation_av_relevancy_max.keys(), tot_rel_av_array_max, alpha=0.5)
			ax.errorbar(simulation_av_relevancy_max.keys(), tot_rel_av_array_max, yerr=tot_rel_error_array_max, fmt="o", label="Entry accuracy: {}".format(assumed_entry_accuracy_effort[1]))
			plt.legend()
			plt.xlabel('Number of actioned entries')
			plt.ylabel('Number of entries with > 0.5 relevance')

			st.pyplot(fig)

			st.subheader('Estimated average relevance vs entries actioned')
			fig, ax = plt.subplots()
			av_rel_av_array = [np.mean(values) for values in simulation_av_relevancy.values()]
			av_rel_error_array = [np.std(values) for values in simulation_av_relevancy.values()]
			av_rel_av_array_max = [np.mean(values) for values in simulation_av_relevancy_max.values()]
			av_rel_error_array_max = [np.std(values) for values in simulation_av_relevancy_max.values()]
			ax.scatter(simulation_av_relevancy.keys(), av_rel_av_array)
			ax.plot(simulation_av_relevancy.keys(), av_rel_av_array, alpha=0.5)
			ax.errorbar(simulation_av_relevancy.keys(), av_rel_av_array, yerr=av_rel_error_array, fmt="o", label="Entry accuracy: {}".format(assumed_entry_accuracy_effort[0]))
			ax.scatter(simulation_av_relevancy_max.keys(), av_rel_av_array_max)
			ax.plot(simulation_av_relevancy_max.keys(), av_rel_av_array_max, alpha=0.5)
			ax.errorbar(simulation_av_relevancy_max.keys(), av_rel_av_array_max, yerr=av_rel_error_array_max, fmt="o", label="Entry accuracy: {}".format(assumed_entry_accuracy_effort[1]))
			plt.legend()
			plt.xlabel('Number of actioned entries')
			plt.ylabel('Estimated average relevance')

			st.pyplot(fig)
else:
	st.write('Please upload a file with your results')





			
