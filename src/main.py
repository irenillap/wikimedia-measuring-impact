import nlp
import wikidata
import mwapi_queries

import pandas as pd
import numpy as np

import datetime
import json

from nltk.corpus import stopwords

def benchmark_values(benchmark_data, final_results):
    """
    Function to benchmark relevant Wikipedia entry values into percentiles based on benchmark data

    Inputs:
    * benchmark_data - dictionary of benchmark images with their page views and page completeness
    * final_results - pd DataFrame of current results of images and candidate Wikipedia entries

    Outputs:
    * pageviewsperc - list of percentile of pageviews for each Wikipedia entry
    * pagecompletenessperc - list of percentile of page completeness for each Wikipedia entry
    """
    pageviewsperc = []
    pagecompletenessperc = []

    for row in final_results.iterrows():
        if row[1]['page_views'] < np.max(benchmark_data.pageviews.values):
            index = np.where(np.sort(benchmark_data_df.pageviews.values) > row[1]['page_views'])[0][0]
            perc = np.sort(benchmark_data.page_views_percentile.values)[index]
            pageviewsperc.append(perc)
        else:
            pageviewsperc.append(100)

        if row[1]['page_completeness'] < np.max(benchmark_data.pagecompleteness.values):
            index = np.where(np.sort(benchmark_data_df.pagecompleteness.values) > row[1]['page_completeness'])[0][0]
            perc = np.sort(benchmark_data.page_completeness_percentile.values)[index]
            pagecompletenessperc.append(perc)
        else:
            pagecompletenessperc.append(100)

    return pageviewsperc, pagecompletenessperc

if __name__ == '__main__':
    """
    Main pipeline used to collect, analyse images and create results of impact measure 
    """
    languages = {'english':{'tokenization':nlp.base_tokenization,
                            'stopwords':stopwords.words('english'),
                            'ner':'flair/ner-english-fast'},
                 'french':{'tokenization':nlp.base_tokenization,
                            'stopwords':stopwords.words('french'),
                            'ner':'flair/ner-french'},
                 'german':{'tokenization':nlp.base_tokenization,
                            'stopwords':stopwords.words('german'),
                            'ner':'flair/ner-german'},
                 'spanish':{'tokenization':nlp.base_tokenization,
                            'stopwords':stopwords.words('spanish'),
                            'ner':'flair/ner-spanish-large'},
                 'dutch':{'tokenization':nlp.base_tokenization,
                            'stopwords':stopwords.words('dutch'),
                            'ner':'flair/ner-dutch'},
                 'chinese':{'tokenization':nlp.chinese_tokenization,
                            'stopwords':None,
                            'ner':None},
                 'japanese':{'tokenization':nlp.japanese_tokenization,
                            'stopwords':None,
                            'ner':None}}
    selected_language = 'english'
    
    tokenizer = languages[selected_language]['tokenization']
    
    stopword = languages[selected_language]['stopwords']
    
    use_ner = True
    
    if use_ner:
        
        try:
            
            ner_model_name = languages[selected_language]['ner']
            
            ner_filter = True
            
            # load tagger
            
            from flair.models import SequenceTagger

            tagger = SequenceTagger.load(ner_model_name)
            
        except:
            
            print('language not supported')
            
            use_ner = False
            
            ner_filter = False
            
            tagger = None
        
    else:
    
        ner_filter = False
        
        tagger = None
        
    with open(r'/content/wikimedia-measuring-impact/src/tokenized_summaries.txt') as f:
        
        tokenized_summaries = eval(f.read())
        
    # Collect images in a (hardcoded for now) collection
    images = wikidata.images_in_collection()

    # Open benchmark data to calculate accurate benchmarks for page views and page completeness
    benchmark_data = json.load(open('benchmark_data.json','r'))
    benchmark_data_df, summaries = mwapi_queries.process_benchmark_data(benchmark_data=benchmark_data)

    # Find where these images are currently being used
    image_usage = {}
    for image in images:
        image_usage[image] = mwapi_queries.image_usage_query(image)

    image_corpus = {}
    final_results = pd.DataFrame()

    # Iterate through all images in collection and find some suitable Wikipedia entries.
    # For each image-Wikipedia entry combination, calculate page views, page completeness,
    # title relevance and image uniqueness measures
    for i, image in enumerate(images):
        print("***********************************************")
        print("Processing image {}, {} out of {}".format(image, i+1, len(images)))
        final_words = nlp.create_image_main_words(
                                              image_title=image,
                                              tokenizer = tokenizer,
                                              stopword = stopword,
                                              nlp_filter='landscape',
                                              ner_filter = ner_filter,
                                              tagger = tagger
                                                  )
        search_results = mwapi_queries.word_search_query_compound(words=final_words,
                                                                  image_title=image)
        image_corpus[image] = mwapi_queries.entry_text_query(pages=search_results)
        if len(search_results) > 0:
            if use_ner:
                tf_idf = nlp.tf_idf(page_summaries={k:nlp.ner_tokenization(v, tokenizer, stopword, tagger, ['PER','LOC','ORG','MISC']) for k,v in image_corpus[image].items()},
                                    training_summaries=tokenized_summaries,
                                    final_words=final_words,
                                    image=image)
            else:
                tf_idf = nlp.tf_idf(page_summaries={k:tokenizer(v, stopword = stopword) for k,v in image_corpus[image].items()},
                                    training_summaries=summaries,
                                    final_words=final_words,
                                    image=image)

            tf_idf.loc[:,'page_views'] = np.tile(None, len(tf_idf))
            tf_idf.loc[:,'page_completeness'] = np.tile(None, len(tf_idf))

            views = {}
            for page in search_results:
                page_views = mwapi_queries.page_views_query(page)
                tf_idf.loc[tf_idf.entry==page,'page_views'] = int(page_views)
                tf_idf.loc[tf_idf.entry==page,'page_completeness'] = mwapi_queries.page_completeness(page=page)
                similar_images = mwapi_queries.image_uniqueness(page=page,main_image=image,relevant_words=final_words)
                tf_idf.loc[tf_idf.entry==page, 'image_uniqueness'] = np.tile(similar_images, len(tf_idf[tf_idf.entry == page]))

        else:
            tf_idf = None
        final_results = final_results.append(tf_idf)

    # Change absolute values of page views and page completeness to relative percentiles based on benchmark data
    pageviewsperc, pagecompletenessperc = benchmark_values(benchmark_data=benchmark_data_df, 
                                                           final_results=final_results)

    final_results['page_views_percentile'] = pageviewsperc
    final_results['page_completeness_percentile'] = pagecompletenessperc

    # Combine all components into one final score
    final_results['final_score'] = 0.25*final_results['word_relevance'] \
                                 + 0.25*final_results['page_views_percentile']*0.01 \
                                 + 0.25*final_results['page_completeness_percentile']*0.01 \
                                 + 0.25*final_results['image_uniqueness']
    final_results.final_score = final_results.final_score.round(2)
    final_results = final_results.reset_index(drop=True)

    # Save results to csv
    final_results.to_csv('final_results.csv',index=False)


