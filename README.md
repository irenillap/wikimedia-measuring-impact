# wikimedia-measuring-impact
Package to measure impact of wikimedia collections

Formed of two main components:
1. main.py - Script to:
	* Collect images in Wikimedia collection
	* Analyse their current usage within Wikipedia entries
	* Using the image titles, find Wikipedia entry candidates and calculate several components (page views, lengths of entry, relevance of text and image uniqueness)
	* Create and locally save a csv with these components for each relevant image and entry

2. app.py - Streamlit app to:
	* Load and visualise results reflected in csv created by script

To run the pipeline:
1. Ensure that all requirements in src/requirements.txt have been installed
2. For initial script, it should be enough to run the following command:
       python main.py
3. To view the results, ensure streamlit is installed. Running the below command should open a local web application:
       streamlit run app.py
