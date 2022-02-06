import mwapi
from mwapi.errors import APIError
from mwviews.api import PageviewsClient

import jellyfish
import numpy as np
import wikipedia
import pandas as pd

def image_usage_query(image):
    """
    Function to retrieve Wikipedia entries where image is being used

    Input:
    * image: string of image title as appearing on Wikimedia

    Output:
    * image_usage: list of relevant Wikipedia entry titles
    """
    session = mwapi.Session(host='https://en.wikipedia.org/',
                            user_agent='measuring_impact/0.0 (irene.iriarte.c@gmail.com)')

    continued = session.get(
        formatversion=2,
        action='query',
        generator='imageusage',
        giutitle=image,
        giulimit=100,  # 100 results per request
        continuation=True)

    image_usage = []
    for portion in continued:
        if 'query' in portion:
            for page in portion['query']['pages']:
                image_usage.append(page['title'])

    print("Image {} appears {} times".format(image, len(image_usage)))

    if len(image_usage) == 0:
        image_usage = ['Image not used']

    return image_usage


def page_views_query(page):
    """
    Function to return average monthly views on page since Jan 2020
    TODO: make this a more dynamic window, possibly determined by specified timeframe inputted by user

    Input:
    * page: Wikipedia page title

    Output:
    * float of monthly average views of page
    """
    p = PageviewsClient(user_agent="measuring_impact/0.0 (irene.iriarte.c@gmail.com)")


    try:
        views = p.article_views('en.wikipedia', 
                                page, 
                                granularity='monthly', 
                                start='20210101')
    except:
        return 0

    if len(views) == 0:
        return 0

    average_since_2020 = []
    for date, view in views.items():
        for page, number in view.items():
            if number is not None:
                average_since_2020.append(number)

    return np.mean(average_since_2020)


def page_completeness(page):
    """
    Function to return the number of words in a Wikipedia page

    Input:
    * page: Wikipedia page title

    Output:
    * int of (approx.) number of words
    """
    try:
        page_info = wikipedia.page(page)

        return len(page_info.content.split(' '))
    
    except:
        return None 


def word_search_query_compound(words, image_title):
    """
    Function to return potential Wikipedia entry candidates for a certain image. If compound words (eg 'Harlem castle') 
    exist in set of relevant words, this will be the main search. Otherwise, the whole title will be searched for instead.

    Input:
    * words - set of relevant words extracted from image title
    * image_title - string of image title as appears on Wikimedia

    Output:
    * image_word_search_results - list of candidate Wikipedia entries returning from searching for relevant words
    """
    image_word_search_results = []

    compound_words = [word for word in words if len(word.split(' ')) > 1]
   
    try:
        if len(compound_words) > 0:
            image_word_search_results = wikipedia.search(compound_words[0], results=10)
            print("There are compound words, searching for {}".format(compound_words[0]))
        else:
            if image_title[-4:] == 'jpeg':
                image_word_search_results = wikipedia.search(image_title[5:-5],results=10)
                print("There are no compound words, searching for {}".format(image_title[5:-5]))
            else:
                image_word_search_results = wikipedia.search(image_title[5:-4],results=10)
                print("There are no compound words, searching for {}".format(image_title[5:-4]))
    except:
        image_word_search_results = []

    return image_word_search_results


def entry_text_query(pages):
    """
    Function to retrieve summaries for all possible pages

    Inputs:
    * pages - list of all considered Wikipedia entries titles

    Outputs:
    * page_corpus - dictionary with all pages as keys and their summaries as values
    """
    page_corpus = {}
    for page in pages:
        try:
            page_corpus[page] = wikipedia.summary(page)
        except:
            print("Warning no summary page for {}".format(page))
            continue

    return page_corpus


def image_uniqueness(page, main_image, relevant_words):
    """
    Function to calculate the image uniqueness component of the impact metric. This is calculated by 
    finding the overlap of relevant words that are present in the relevant image title and the image titles
    of other images in the Wikipedia entry. Also checks whether image is already in Wikipedia entry.

    Inputs:
    * page - string of Wikipedia entry title
    * main_image - string of title of image in collection
    * relevant_words - set of relevant words extracted from title

    Outputs:
    * float of image uniqueness value for Wikipedia entry and image combination
    """
    try:
        page_info = wikipedia.page(page)
        images = page_info.images
        total_overlap = []

        for image in images:
            image = image.split('/')[-1].lower()
            if image == main_image[5:]:
                return 'Image in entry'
            image_overlap = []
            for word in relevant_words:
                if word in image:
                    image_overlap.append(1)
                else:
                    image_overlap.append(0)
            
            total_overlap.append(sum(image_overlap)/float(len(relevant_words)))

        final_value = sum([1 for overlap in total_overlap if overlap >= 0.5])
        if final_value == 0:
            return 1.0
        else:
            return 1.0 / final_value
    except:
        print("Warning no extra images for {}".format(page))
        return 0.5


def process_benchmark_data(benchmark_data):
    """
    Function to process benchmark data including calculating percentiles for page views and page completeness

    Inputs:
    * benchmark_data - dictionary of benchmark data

    Outputs:
    * benchmark_df - pd DataFrame of benchmark data titles with their page views and page completeness percentiles 
    * summaries - dictionary of each entry and its summary
    """
    entries = []
    pageviews = []
    pagecompleteness = []
    summaries = {}

    for entry, data in benchmark_data.items():
        entries.append(entry)
        pageviews.append(data['views'])
        pagecompleteness.append(data['completeness'])
        summaries[entry] = data['summary']

    benchmark_df = pd.DataFrame(index=entries)
    benchmark_df['pageviews'] = pageviews
    benchmark_df['pagecompleteness'] = pagecompleteness

    benchmark_df['page_views_percentile'] = pd.qcut(benchmark_df['pageviews'].values, 100, labels=False, duplicates='drop')
    benchmark_df['page_completeness_percentile'] = pd.qcut(benchmark_df['pagecompleteness'].values, 100, labels=False, duplicates='drop')

    return benchmark_df, summaries

