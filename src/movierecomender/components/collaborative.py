# collaborative.py
import pandas as pd
import logging

class CollaborativeFiltering:
    def __init__(self, user_movie_matrix):
        self.user_movie_matrix = user_movie_matrix

    def get_recommendations(self, user_id, top_n=10):
        """Get collaborative filtering recommendations for a given user."""
        if user_id not in self.user_movie_matrix.index:
            logging.error(f"User ID '{user_id}' not found in the user-movie matrix.")
            return pd.DataFrame(columns=['movieId', 'title', 'rating'])

        # Calculate user similarity
        logging.info(f"Calculating recommendations for user ID {user_id}...")
        user_similarity = self.user_movie_matrix.corrwith(self.user_movie_matrix.loc[user_id])
        similar_users = user_similarity.dropna().sort_values(ascending=False).iloc[1:top_n+1]  # Exclude self-similarity

        # Filter similar users to ensure they are in the user_movie_matrix
        valid_similar_users = similar_users.index.intersection(self.user_movie_matrix.index)

        # Get movie recommendations based on the mean ratings from similar users
        recommendations = self.user_movie_matrix.loc[valid_similar_users].mean().sort_values(ascending=False).head(top_n)
        recommendations_df = recommendations.reset_index()
        recommendations_df.columns = ['movieId', 'rating']
        
        return recommendations_df
