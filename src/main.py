import nlp
import wikidata
import mwapi_queries

import pandas as pd
import numpy as np

import datetime


if __name__ == '__main__':
    """
    Main pipeline used to collect, analyse images and create results of impact measure 
    """
    # Collect images in a (hardcoded for now) collection
    images = wikidata.images_in_collection()

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
        final_words = nlp.create_image_main_words(image_title=image,
                                                  nlp_filter='landscape')
        search_results = mwapi_queries.word_search_query(words=final_words,
                                                         image_title=image)
        image_corpus[image] = mwapi_queries.entry_text_query(pages=search_results)
        if len(search_results) > 0:
            tf_idf = nlp.tf_idf(page_summaries=image_corpus[image],
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

    # Change absolute values of page views and page completeness to relative percentiles
    final_results['page_views_percentile'] = pd.qcut(final_results['page_views'].values, 100, labels=False, duplicates='drop')
    final_results['page_completeness_percentile'] = pd.qcut(final_results['page_completeness'].values, 100, labels=False, duplicates='drop')

    # Combine all components into one final score
    final_results['final_score'] = 0.25*final_results['word_relevance'] \
                                 + 0.25*final_results['page_views_percentile']*0.01 \
                                 + 0.25*final_results['page_completeness_percentile']*0.01 \
                                 + 0.25*final_results['image_uniqueness']
    final_results.final_score = final_results.final_score.round(2)
    final_results = final_results.reset_index(drop=True)

    # Save results to csv
    final_results.to_csv('final_results.csv',index=False)


