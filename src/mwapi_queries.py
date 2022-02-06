import mwapi
from mwapi.errors import APIError
from mwviews.api import PageviewsClient

import jellyfish
import numpy as np
import wikipedia
import pandas as pd

def image_usage_query(image):
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
    try:
        page_info = wikipedia.page(page)

        return len(page_info.content.split(' '))
    
    except:
        return None 


def word_search_query(words, image_title, relevance_limit=50):
    session = mwapi.Session(host='https://en.wikipedia.org/',
                            user_agent='measuring_impact/0.0 (irene.iriarte.c@gmail.com)')

    image_word_search_results = []

    for word in words:
        continued = session.get(
            formatversion=2,
            action='query',
            generator='search',
            gsrsearch=word,
            gsrlimit=relevance_limit + 1,
            continuation=True)

        for portion in continued:
            if 'query' in portion:
                word_results = len(portion['query']['pages'])
                if word_results > relevance_limit:
                    print("Word {} has over {} results, so will be discounted".format(word, word_results))
                    break
                else:
                    print("Word {} has {} results, so will be used for relevancy".format(word, word_results))
                    for page in portion['query']['pages']:
                        image_word_search_results.append(page['title'])

    if len(image_word_search_results) == 0:
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

def word_search_query_compound(words, image_title):
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
    page_corpus = {}
    for page in pages:
        try:
            page_corpus[page] = wikipedia.summary(page)
        except:
            print("Warning no summary page for {}".format(page))
            continue

    return page_corpus


def image_uniqueness(page, main_image, relevant_words):
    try:
        page_info = wikipedia.page(page)
        images = page_info.images
        total_overlap = []
        # print(main_image)
        for image in images:
            image = image.split('/')[-1].lower()
            # print(image)
            if image == main_image[5:]:
                return 'Image in entry'
            image_overlap = []
            for word in relevant_words:
                # import pdb
                # pdb.set_trace()
                if word in image:
                    image_overlap.append(1)
                else:
                    image_overlap.append(0)
            
            # print(sum(image_overlap)/float(len(relevant_words)))
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

