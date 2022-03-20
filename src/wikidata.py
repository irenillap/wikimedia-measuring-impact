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
    LIMIT 20
    """

    r = requests.get(url, params={'format': 'json',
                                  'query': query})

    data = r.json()

    images = []
    for item in data['results']['bindings']:
        images.append(item['fileTitle']['value'])

    print("Fetched {} images".format(len(images)))

    return images


def images_in_collection(collection_wikipedia_id, retrieval_limit=25):
    """
    Function to return 200 images on Wikimedia tagged as belonging to a certain collection

    Input:
    * collection_wikipedia_id: string of collection id from wikimedia
    * retrieval_limit: int of limit of retrieval

    Output:
    * images: dictionary
    """
    url = 'https://query.wikidata.org/sparql'


    query = """
    SELECT DISTINCT *
    WHERE {{
        ?item wdt:P195 wd:{}.
        ?item wdt:P18 ?image.
        BIND(CONCAT("File:", STRAFTER(wikibase:decodeUri(STR(?image)), "http://commons.wikimedia.org/wiki/Special:FilePath/")) AS ?fileTitle)
    }}
    LIMIT {}
    """.format(collection_wikipedia_id, retrieval_limit)

    r = requests.get(url, params={'format': 'json',
                                  'query': query})

    data = r.json()

    images = []
    for item in data['results']['bindings']:
        images.append(item['fileTitle']['value'])

    print("Fetched {} images".format(len(images)))

    return images


def images_in_portrait_collection():
    """
    Function to return 200 images on Wikimedia tagged as belonging to a certain collection

    Input:
    * collection_wikipedia_id: string of collection id from wikimedia
    * retrieval_limit: int of limit of retrieval

    Output:
    * images: dictionary
    """
    url = 'https://query.wikidata.org/sparql'


    query = """
    SELECT DISTINCT ?item ?image ?fileTitle ?sitter ?sitterLabel
    WHERE {
        ?item wdt:P195 wd:Q54859927.
        ?item wdt:P18 ?image.
        BIND(CONCAT("File:", STRAFTER(wikibase:decodeUri(STR(?image)), "http://commons.wikimedia.org/wiki/Special:FilePath/")) AS ?fileTitle)
        ?item wdt:P921 ?sitter.
        
              
        SERVICE wikibase:label {
       bd:serviceParam wikibase:language "en" 
    }}
     ORDER BY ASC(?item)
     LIMIT 50
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


