import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def content_based_recommendation(data, item_name, top_n=5):

    if item_name not in data['ProductName'].values:
        print("Product not found.")
        return pd.DataFrame()

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['Tags'])

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    item_index = data[data['ProductName'] == item_name].index[0]

    similarity_scores = list(enumerate(similarity_matrix[item_index]))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similarity_scores = similarity_scores[1:top_n+1]

    product_indices = [i[0] for i in similarity_scores]

    return data.iloc[product_indices][['ProductName', 'Brand', 'ReviewCount']]

    def content_based_recommendation(data, item_name, top_n=5):
    ...
    return recommended_items


# ADD THIS AT THE VERY BOTTOM

if __name__ == "__main__":
    import pandas as pd
    from cleaning_data import process_data

    data = pd.read_csv("clean_data.csv")
    data = process_data(data)

    print("Testing Content-Based Recommendation...\n")

    result = content_based_recommendation(
        data,
        item_name=data['ProductName'].iloc[0],
        top_n=5
    )

    print(result)