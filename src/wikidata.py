import requests
import numpy as np

def images_owned_by():
    """
    Function to return all images on Wikimedia tagged as owned by a certain institution
    (currently all hardcoded)
    POSSIBLE TODO: change to not being hardcoded

    Input:

    Output:
    * images: dictionary
    """
    url = 'https://query.wikidata.org/sparql'

    query = """
    SELECT DISTINCT *
    WHERE {
        ?collection wdt:P127 wd:Q666063.
        ?item wdt:P195 ?collection.
        ?item wdt:P18 ?image.
        BIND(CONCAT("File:", STRAFTER(wikibase:decodeUri(STR(?image)), "http://commons.wikimedia.org/wiki/Special:FilePath/")) AS ?fileTitle)
      }
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
    """
    Function to return 200 images on Wikimedia tagged as belonging to a certain collection
    (currently all hardcoded)
    POSSIBLE TODO: change to collection and limit not being hardcoded

    Input:

    Output:
    * images: dictionary
    """
    url = 'https://query.wikidata.org/sparql'

    query = """
    SELECT DISTINCT *
    WHERE {
        ?item wdt:P195 wd:Q21542493.
        ?item wdt:P18 ?image.
        BIND(CONCAT("File:", STRAFTER(wikibase:decodeUri(STR(?image)), "http://commons.wikimedia.org/wiki/Special:FilePath/")) AS ?fileTitle)
      }
    LIMIT 200
    """

    r = requests.get(url, params={'format': 'json',
                                  'query': query})

    data = r.json()

    images = []
    for item in data['results']['bindings']:
        images.append(item['fileTitle']['value'])

    print("Fetched {} images".format(len(images)))

    return images


def process_pilot_collection():
    """
    Function to return and process 200 images on Wikimedia tagged as belonging to the NLW landscape collection
    (currently all hardcoded)
    POSSIBLE TODO: change to collection and limit not being hardcoded

    Input:

    Output:
    * images: dictionary
    """
    url = 'https://query.wikidata.org/sparql'

    query = """
    SELECT DISTINCT *
    WHERE {
        ?item wdt:P195 wd:Q21542493.
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


