import mwapi
from mwapi.errors import APIError
from mwviews.api import PageviewsClient

import wikipedia

import numpy as np

import json

# Retrieve random pages
session = mwapi.Session(host='https://en.wikipedia.org/',
                        user_agent='measuring_impact/0.0 (irene.iriarte.c@gmail.com)')

continued = session.get(
    formatversion=2,
    action='query',
    list='random',
    rnnamespace=0,
    rnlimit=1,
    continuation=True)

i=0
random_pages = []
for portion in continued:
    if 'query' in portion:
        for page in portion['query']['random']:
            random_pages.append(page['title'])
            print(page['title'])
            i+=1
        if i > 1000:
            break

print('*****************************************************')

# Find average pageviews for those random pages
p = PageviewsClient(user_agent="measuring_impact/0.0 (irene.iriarte.c@gmail.com)")

benchmark_data = {}
i = 0
for page in random_pages:
    benchmark_data[page] = {}
    try:
        views = p.article_views('en.wikipedia', 
                                page, 
                                granularity='monthly', 
                                start='20210101')

        average_since_2020 = []
        for date, view in views.items():
            for article, number in view.items():
                if number is not None:
                    average_since_2020.append(number)

        benchmark_data[page]['views'] = np.mean(average_since_2020)
        
    except:
        benchmark_data[page]['views'] = 0
    print(page, benchmark_data[page]['views'])
    i+=1

print('*****************************************************')
# Find length for those random pages
for page in random_pages:
    try:
        page_info = wikipedia.page(page)

        benchmark_data[page]['completeness'] = len(page_info.content.split(' '))
        benchmark_data[page]['summary'] = wikipedia.summary(page)
    
    except:
        benchmark_data[page]['completeness'] = 0
        benchmark_data[page]['summary'] = ''
    print(page, benchmark_data[page]['completeness'])

print('*****************************************************')
# for page in random_pages:
#     print('Retrieving summary for',page)
#     try:
#         page_summary = wikipedia.summary(page)

#         benchmark_data[page]['summary'] = page_summary
    
#     except:
#         benchmark_data[page]['summary'] = ''

# print('*****************************************************')
json.dump(benchmark_data, open('benchmark_data.json','w'))
