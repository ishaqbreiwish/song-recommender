import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

# Class for Popularity-Based Recommender System
class PopularityRecommender:
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None

    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns={self.user_id: 'score'}, inplace=True)

        # Sort songs by recommendation score and item_id
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[0, 1])

        # Generate recommendation rank based on score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        # Store the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations.copy()
        user_recommendations['user_id'] = user_id

        # Reorder columns to bring user_id to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        return user_recommendations

# Class for Item Similarity-Based Recommender System
class ItemSimilarityRecommender:
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.co_matrix = None

    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # Build co-occurrence matrix
        user_item_matrix = train_data.pivot_table(index=user_id, columns=item_id, aggfunc='size', fill_value=0)
        self.co_matrix = user_item_matrix.T.dot(user_item_matrix)
        np.fill_diagonal(self.co_matrix.values, 0)

    def get_user_items(self, user_id):
        user_data = self.train_data[self.train_data[self.user_id] == user_id]
        return list(user_data[self.item_id].unique())

    def recommend(self, user_id, top_n=10):
        user_items = self.get_user_items(user_id)
        item_scores = {}

        for item in user_items:
            if item in self.co_matrix.index:
                similar_items = self.co_matrix.loc[item]
                for similar_item, score in similar_items.items():
                    if similar_item not in user_items:
                        item_scores[similar_item] = item_scores.get(similar_item, 0) + score

        recommendations = (
            pd.DataFrame.from_dict(item_scores, orient='index', columns=['score'])
            .sort_values(by='score', ascending=False)
            .head(top_n)
            .reset_index()
            .rename(columns={'index': 'song'})
        )
        return recommendations


def compute_svd(urm, k):
    U, s, Vt = svds(urm, k=k)
    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i, i] = np.sqrt(s[i])
    return csc_matrix(U, dtype=np.float32), csc_matrix(S, dtype=np.float32), csc_matrix(Vt, dtype=np.float32)

def compute_estimated_ratings(urm, U, S, Vt, u_test, k):
    right_term = S @ Vt
    estimated_ratings = np.zeros(shape=(urm.shape[0], urm.shape[1]), dtype=np.float16)
    for user in u_test:
        prod = U[user, :] @ right_term
        estimated_ratings[user, :] = prod
        recommendations = (-estimated_ratings[user, :]).argsort()[:k]
    return recommendations

# Main Function
if __name__ == "__main__":
    # Load data
    triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
    songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'

    song_df_1 = pd.read_table(triplets_file, header=None)
    song_df_1.columns = ['user_id', 'song_id', 'listen_count']

    song_df_2 = pd.read_csv(songs_metadata_file)
    song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

    # Split data
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(song_df, test_size=0.20, random_state=0)

    # Popularity Model
    pm = PopularityRecommender()
    pm.create(train_data, 'user_id', 'song')
    print("Popularity-Based Recommendations:")
    print(pm.recommend(user_id=1))

    # Item Similarity Model
    is_model = ItemSimilarityRecommender()
    is_model.create(train_data, 'user_id', 'song')
    print("\nPersonalized Recommendations:")
    print(is_model.recommend(user_id=1))

    # SVD Model
    urm = csc_matrix((train_data['listen_count'], (train_data['user_id'], train_data['song_id'])), dtype=np.float32)
    U, S, Vt = compute_svd(urm, k=2)
    u_test = [1]
    print("\nSVD-Based Recommendations:")
    print(compute_estimated_ratings(urm, U, S, Vt, u_test, k=5))
