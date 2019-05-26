import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

RECOMMENDATION_PATH = os.path.join("datasets", "recommendation")




# Returns CSV data to Pandas DataFrame
def load_recommendation_data(recommendation_path=RECOMMENDATION_PATH):
    csv_path = os.path.join(recommendation_path, "dataset_catalog.csv")
    return pd.read_csv(csv_path)


def load_recommendation_events_data(recommendation_path=RECOMMENDATION_PATH):
    csv_path = os.path.join(recommendation_path, "dataset_events.csv")
    return pd.read_csv(csv_path)

def prepare_data(query_id):

    dataset_catalog = load_recommendation_data()
    dataset_events = load_recommendation_events_data()

    dataset = pd.merge(dataset_catalog, dataset_events, on='product_id')

    # converts categories to numerical values
    cat = pd.Categorical(dataset.type,
                         categories=['view_product', 'add_to_cart', 'purchase_item'],
                         ordered=True)
    labels = pd.factorize(cat, sort=True)[0] + 1
    dataset.type = labels

    category_interactions = dataset[dataset.category_id == 1].groupby(by=['product_id'])[
        'type'].sum().reset_index().rename(
        columns={'type': 'total_interactions'})

    # Constructs dataset with customer and customer interaction count
    customers_dataset = dataset.groupby(by=['customer_id', 'product_id', 'category_id'])[
        'type'].sum().reset_index().rename(
        columns={'type': 'customer_interactions'})

    total_interactions = customers_dataset.groupby(by=['product_id'])[
        'customer_interactions'].sum().reset_index().rename(
        columns={'customer_interactions': 'total_interactions'})

    # total_interactions
    dataset_with_interactions = customers_dataset.merge(total_interactions, left_on='product_id', right_on='product_id',
                                                        how='left')

    cicina = (dataset_with_interactions[dataset_with_interactions['total_interactions'] > 450])

    print(dataset_with_interactions.info())

    # dataset with only category_id == 1
    category_one_dataset = dataset_with_interactions.query('category_id == \'1\'')


    wide_dataset_cat = dataset_with_interactions.pivot(index='product_id', columns='customer_id',
                                                       values='customer_interactions').fillna(0)

    wide_dataset_cat_sparse = csr_matrix(wide_dataset_cat.values)

    user_id_index = dataset_with_interactions[dataset_with_interactions['customer_id']==query_id].index.values.astype(int)[0]
    print(user_id_index)
    return wide_dataset_cat_sparse, wide_dataset_cat, user_id_index


def get_recommendation(query_id):

    wide_dataset_cat_sparse, wide_dataset_cat, user_id_index = prepare_data(query_id)
    print(wide_dataset_cat)

    model_kmm = NearestNeighbors(metric='cosine', algorithm='brute')
    model_kmm.fit(wide_dataset_cat_sparse)

    # chooses random index from all users
    query_index = np.random.choice(wide_dataset_cat.shape[0])

    distances, indices = model_kmm.kneighbors(wide_dataset_cat.iloc[user_id_index, :].values.reshape(1, -1), n_neighbors=6)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(wide_dataset_cat.index[user_id_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, wide_dataset_cat.index[indices.flatten()[i]],
                                                           distances.flatten()[i]))
get_recommendation(5)