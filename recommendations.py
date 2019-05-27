import os
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import json

# local path for datasets
RECOMMENDATION_PATH = os.path.join("datasets", "recommendation")
popularity_threshold = 1200

# Loads CSV data to Pandas DataFrame
def load_recommendation_data(recommendation_path=RECOMMENDATION_PATH):
    csv_path = os.path.join(recommendation_path, "dataset_catalog.csv")
    return pd.read_csv(csv_path)

# Loads CSV data to Pandas DataFrame
def load_recommendation_events_data(recommendation_path=RECOMMENDATION_PATH):
    csv_path = os.path.join(recommendation_path, "dataset_events.csv")
    return pd.read_csv(csv_path)

# Function for preparing data from CSV
def prepare_data(query_id):

    dataset_catalog = load_recommendation_data()
    dataset_events = load_recommendation_events_data()

    dataset = pd.merge(dataset_catalog, dataset_events, on='product_id')

    # Converts categorical data to to numerical values
    cat = pd.Categorical(dataset.type,
                         categories=['view_product', 'add_to_cart', 'purchase_item'],
                         ordered=True)
    labels = pd.factorize(cat, sort=True)[0] + 1
    dataset.type = labels

    # Constructs dataset with customer and customer interaction count
    customers_dataset = dataset.groupby(by=['customer_id', 'product_id', 'category_id'])[
        'type'].sum().reset_index().rename(
        columns={'type': 'customer_interactions'})

    # Adds total interactions column, I thought it was useful but probably isn't :D
    total_interactions = customers_dataset.groupby(by=['product_id'])[
        'customer_interactions'].sum().reset_index().rename(
        columns={'customer_interactions': 'total_interactions'})

    # Merges dataset with total_interactions column
    dataset_with_interactions = customers_dataset.merge(total_interactions, left_on='product_id', right_on='product_id',
                                                        how='left')

    popular_dataset = dataset_with_interactions.query('total_interactions >= @popularity_threshold')

    popular_dataset = popular_dataset.append(dataset_with_interactions.loc[dataset_with_interactions['customer_id'] == query_id])

    if not popular_dataset[popular_dataset.duplicated(['customer_id', 'product_id'])].empty:
        initial_rows = popular_dataset.shape[0]

        print('Initial dataframe shape {0}'.format(popular_dataset.shape))
        popular_dataset = popular_dataset.drop_duplicates(['customer_id', 'product_id'])
        current_rows = popular_dataset.shape[0]
        print('New dataframe shape {0}'.format(popular_dataset.shape))
        print('Removed {0} rows'.format(initial_rows - current_rows))


    wide_dataset_cat = popular_dataset.pivot(index='product_id', columns='customer_id', values='customer_interactions').fillna(0)

    wide_dataset_cat_sparse = csr_matrix(wide_dataset_cat.values)

    user_id_index = 0
    for index, val in enumerate(wide_dataset_cat):
        if val == query_id:
            user_id_index = index
    return wide_dataset_cat_sparse, wide_dataset_cat, user_id_index


def get_recommendation(query_index):

    wide_dataset_cat_sparse, wide_dataset_cat, user_id_index = prepare_data(query_index)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(wide_dataset_cat_sparse)

    distances, indices = model_knn.kneighbors(wide_dataset_cat.iloc[user_id_index, :].values.reshape(1, -1), n_neighbors=6)

    list = []
    for i in range(1, len(distances.flatten())):
        list.append({"product_id": int(wide_dataset_cat.index[indices.flatten()[i]]), "distance": (distances.flatten()[i])})

    result_data = {
        "customer_id": int(wide_dataset_cat.index[user_id_index]),
        "results": list
    }
    print(result_data)
    return json.dumps(result_data)
