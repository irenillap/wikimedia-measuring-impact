import requests
import numpy as np

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
