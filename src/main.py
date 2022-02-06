import nlp
import wikidata
import mwapi_queries

import pandas as pd
import numpy as np

import datetime


def pipeline():
    images = wikidata.images_owned_by()
    
    # image_usage = {}
    # for image in images:
    #     image_usage[image] = mwapi_queries.image_usage_query(image)

    import json
    image_usage = json.load(open('image_usage.json','r'))

    create_collection_main_words(images=images)

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
            page_views[image] =  mwapi_queries.page_views_query(pages)

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
    return image_usage_df


if __name__ == '__main__':
    images = wikidata.images_in_collection()
    import pdb
    pdb.set_trace()
    # import json
    # image_usage = json.load(open('image_usage.json','r'))

    image_usage = {}
    for image in images:
        image_usage[image] = mwapi_queries.image_usage_query(image)

    image_corpus = {}
    final_results = pd.DataFrame()

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

    final_results['page_views_percentile'] = pd.qcut(final_results['page_views'].values, 100, labels=False, duplicates='drop')
    final_results['page_completeness_percentile'] = pd.qcut(final_results['page_completeness'].values, 100, labels=False, duplicates='drop')

    final_results['final_score'] = 0.25*final_results['word_relevance'] \
                                 + 0.25*final_results['page_views_percentile']*0.01 \
                                 + 0.25*final_results['page_completeness_percentile']*0.01 \
                                 + 0.25*final_results['image_uniqueness']
    final_results.final_score = final_results.final_score.round(2)
    final_results = final_results.reset_index(drop=True)

    final_results.to_csv('final_results.csv',index=False)


