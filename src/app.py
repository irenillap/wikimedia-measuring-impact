import streamlit as st
import pandas as pd
import numpy as np

import main

st.title('Wikipedia Images User Visits')

@st.cache
def load_data():
	image_usage_df = main.pipeline()
	return image_usage_df

data_load_state = st.text('Loading data...')
image_usage_df = load_data()
data_load_state.text('Loading data...done!')

st.subheader('Raw data')
st.write(image_usage_df)

selected_indices = st.multiselect('Select images:', image_usage_df.image)
st.write(selected_indices)
selected_rows = image_usage_df[image_usage_df.image.isin(selected_indices)]

st.subheader('Chart of image visits')
linedata = pd.DataFrame(selected_rows[selected_rows.columns.values[3:-1]]).T
st.line_chart(linedata)