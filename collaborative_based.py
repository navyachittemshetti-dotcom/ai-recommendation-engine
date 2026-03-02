import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_filtering(data, user_id, top_n=5):

    # Check if user exists
    if user_id not in data['UserID'].values:
        print(f"User {user_id} not found.")
        return pd.DataFrame()

    # Create User-Item Matrix
    user_item_matrix = data.pivot_table(
        index='UserID',
        columns='ProductID',
        values='Rating',
        fill_value=0
    )

    # Calculate similarity between users
    user_similarity = cosine_similarity(user_item_matrix)

    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    # Get similarity scores of target user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # Remove target user itself
    similar_users = similar_users.drop(user_id)

    recommended_products = []

    for sim_user in similar_users.index:
        sim_user_products = data[data['UserID'] == sim_user]

        for _, row in sim_user_products.iterrows():
            if row['ProductID'] not in data[data['UserID'] == user_id]['ProductID'].values:
                recommended_products.append(row)

    # Remove duplicates
    recommended_df = pd.DataFrame(recommended_products).drop_duplicates(subset=['ProductID'])

    return recommended_df[['ProductName', 'Brand', 'Rating']].head(top_n)