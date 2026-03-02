import pandas as pd

def rating_based_recommendation(data, top_n=5):

    # Calculate average rating for each product
    rating_avg = data.groupby('ProductID').agg({
        'Rating': 'mean',
        'ProductName': 'first',
        'Brand': 'first'
    }).reset_index()

    # Sort by highest rating
    rating_avg = rating_avg.sort_values(by='Rating', ascending=False)

    return rating_avg[['ProductName', 'Brand', 'Rating']].head(top_n)