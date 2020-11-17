import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

import main

st.title('Wikipedia Images User Visits')

@st.cache
def load_data():
	image_usage_df = main.pipeline()
	return image_usage_df

def get_image(filename):
	import urllib
	parsed_filename = urllib.parse.quote(filename).replace(' ', '_')
	urllib.request.urlretrieve("http://commons.wikimedia.org/wiki/Special:Redirect/file/"+parsed_filename, os.path.join('images/','file_'+filename))

	return os.path.join('images/','file_'+filename)

data_load_state = st.text('Loading data...')
image_usage_df = load_data()
data_load_state.text('Loading data...done!')

st.subheader('Raw data')
st.write(image_usage_df)

selected_indices = st.multiselect('Select images:', image_usage_df.image.unique())
st.write(selected_indices)
selected_rows = pd.DataFrame(image_usage_df[image_usage_df.image.isin(selected_indices)])
st.write(selected_rows)

for image in selected_indices:
	filepath = get_image(image)
	image_obj = Image.open(filepath)
	st.image(image_obj, width=500, caption=image)


selected_pages = st.multiselect('Select pages:', selected_rows.pages.unique())
st.write(selected_pages)
selected_page_rows = pd.DataFrame(selected_rows[selected_rows.pages.isin(selected_pages)])

selected_page_rows = selected_page_rows.reset_index(drop=True)
new_index = []
for row in selected_page_rows.iterrows(): 
	new_index.append(row[1][0] +"|"+ row[1][1])
selected_page_rows.index = new_index

st.subheader('Chart of image visits')
linedata = selected_page_rows[selected_page_rows.columns.values].T[3:-1]
st.line_chart(linedata)