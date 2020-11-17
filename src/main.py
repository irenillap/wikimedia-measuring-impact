import mwapi
from mwapi.errors import APIError
from mwviews.api import PageviewsClient

import requests
import pandas as pd
import numpy as np

import datetime


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


def images_owned_by():
    url = 'https://query.wikidata.org/sparql'

    query = """
    SELECT DISTINCT *
    WHERE {
        ?collection wdt:P127 wd:Q666063.
        ?item wdt:P195 ?collection.
        ?item wdt:P18 ?image.
        BIND(CONCAT("File:", STRAFTER(wikibase:decodeUri(STR(?image)), "http://commons.wikimedia.org/wiki/Special:FilePath/")) AS ?fileTitle)
      }
    LIMIT 300
    """

    r = requests.get(url, params={'format': 'json',
                                  'query': query})

    data = r.json()

    images = []
    for item in data['results']['bindings']:
        images.append(item['fileTitle']['value'])

    print("Fetched {} images".format(len(images)))

    return images


def images_in_collection():
    url = 'https://query.wikidata.org/sparql'

    query = """
    SELECT DISTINCT *
    WHERE {
        ?item wdt:P195 wd:Q23817605.
        ?item wdt:P18 ?image.
        BIND(CONCAT("File:", STRAFTER(wikibase:decodeUri(STR(?image)), "http://commons.wikimedia.org/wiki/Special:FilePath/")) AS ?fileTitle)
      }
    LIMIT 100
    """

    r = requests.get(url, params={'format': 'json',
                                  'query': query})

    data = r.json()

    images = []
    for item in data['results']['bindings']:
        images.append(item['fileTitle']['value'])

    print("Fetched {} images".format(len(images)))

    return images


def collections_owned_by():
    url = 'https://query.wikidata.org/sparql'

    query = """
    SELECT DISTINCT ?collectionLabel
    WHERE {
        ?collection wdt:P127 wd:Q666063.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
      }
    """

    r = requests.get(url, params={'format': 'json',
                                  'query': query})

    data = r.json()


def page_views_query(page):
    p = PageviewsClient(user_agent="measuring_impact/0.0 (irene.iriarte.c@gmail.com)")

    views = p.article_views('en.wikipedia', 
                            page, 
                            granularity='monthly', 
                            start='20200101', 
                            end='20201001')
    
    return views


def pipeline():
    images = images_owned_by()
    
    # image_usage = {}
    # for image in images:
    #     image_usage[image] = image_usage_query(image)

    import json
    image_usage = json.load(open('image_usage.json','r'))

    images = []
    pages = []
    for image, page in image_usage.items():
        images.append(image)
        pages.append(page)

    image_usage_df = pd.DataFrame()
    image_usage_df['image'] = images
    image_usage_df['pages'] = pages
    image_usage_df = image_usage_df.explode('pages')

    page_views = {}
    for image, pages in image_usage.items():
        if pages != ['Image not used']:
            page_views[image] =  page_views_query(pages)

    page_view_keys = [k  for  k in  page_views.keys()]
    dates = [k for k in page_views[page_view_keys[0]].keys()]

    dates_data = {}
    for date in np.sort(dates):
        dates_data[date] = []
        for entry in image_usage_df.iterrows():
            if entry[1][1] != 'Image not used':
                dates_data[date].append(page_views[entry[1][0]][date][entry[1][1].replace(' ','_')])
            else:
                dates_data[date].append(None)
        image_usage_df[date] = dates_data[date]

    image_usage_agg = image_usage_df.groupby('image').sum()

    return image_usage_df


if __name__ == '__main__':
    images = images_owned_by()
    
    # image_usage = {}
    # for image in images:
    #     image_usage[image] = image_usage_query(image)

    import json
    image_usage = json.load(open('image_usage.json','r'))

    images = []
    pages = []
    for image, page in image_usage.items():
        images.append(image)
        pages.append(page)

    image_usage_df = pd.DataFrame()
    image_usage_df['image'] = images
    image_usage_df['pages'] = pages
    image_usage_df = image_usage_df.explode('pages')

    page_views = {}
    for image, pages in image_usage.items():
        if pages != ['Image not used']:
            page_views[image] =  page_views_query(pages)

    page_view_keys = [k  for  k in  page_views.keys()]
    dates = [k for k in page_views[page_view_keys[0]].keys()]

    dates_data = {}
    for date in np.sort(dates):
        dates_data[date] = []
        for entry in image_usage_df.iterrows():
            if entry[1][1] != 'Image not used':
                dates_data[date].append(page_views[entry[1][0]][date][entry[1][1].replace(' ','_')])
            else:
                dates_data[date].append(None)
        image_usage_df[date] = dates_data[date]

    image_usage_agg = image_usage_df.groupby('image').sum()

    import pdb
    pdb.set_trace()


