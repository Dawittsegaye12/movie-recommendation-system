import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class HybridRecommendationSystem:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.cosine_sim = None
        self.user_movie_matrix = None
    
    def load_data(self, train_data_path, test_data_path):
        """Loads training and testing data from pickle files."""
        with open(train_data_path, 'rb') as f:
            self.train_data = pickle.load(f)
        with open(test_data_path, 'rb') as f:
            self.test_data = pickle.load(f)
        print("Data loaded successfully.")

    def preprocess_data(self):
        """Prepares data for recommendations by creating user-item matrices."""
        self.user_movie_matrix = self.train_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        print("User-Item Matrix created successfully.")

        # Prepare movie features for content-based filtering
        self.train_data['genres'] = self.train_data['genres'].astype(str)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.train_data['genres'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("Content similarity matrix computed successfully.")

    def get_recommendations_content_based(self, title, top_n=10):
        """Gets content-based recommendations for a given movie title."""
        idx = self.train_data[self.train_data['title'] == title].index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:top_n + 1]]  # Exclude the first entry (itself)
        return self.train_data.iloc[movie_indices]

    def get_recommendations_collaborative(self, user_id, top_n=10):
        """Gets collaborative filtering recommendations for a given user."""
        user_ratings = self.user_movie_matrix.loc[user_id]
        similar_users = self.user_movie_matrix.corrwith(user_ratings)
        similar_users = similar_users.sort_values(ascending=False).head(top_n)
        recommendations = self.user_movie_matrix.loc[similar_users.index].mean().sort_values(ascending=False)
        recommendations = recommendations[~recommendations.index.isin(user_ratings[user_ratings > 0].index)]
        return recommendations.head(top_n)

    def get_hybrid_recommendations(self, title, user_id, top_n=10):
        """Combines content-based and collaborative filtering recommendations."""
        content_recommendations = self.get_recommendations_content_based(title, top_n)
        collaborative_recommendations = self.get_recommendations_collaborative(user_id, top_n)

        # Combine recommendations
        combined_recommendations = pd.concat([content_recommendations, collaborative_recommendations])
        hybrid_recommendations = combined_recommendations.groupby('title').mean().sort_values('rating', ascending=False)
        return hybrid_recommendations.head(top_n)

    def save_model(self, file_path):
        """Saves the trained model to a pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")

if __name__ == '__main__':
    # Define paths for training, testing, and model saving
    train_data_path = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\artifacts\training_data.pkl'
    test_data_path = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\artifacts\testing_data.pkl'
    model_save_path = os.path.join(os.getcwd(), "hybrid_recommendation_model.pkl")

    # Initialize the hybrid recommendation system
    hybrid_model = HybridRecommendationSystem()
    
    # Load data
    hybrid_model.load_data(train_data_path, test_data_path)
    
    # Preprocess data
    hybrid_model.preprocess_data()
    
    # Example usage
    title = "The Matrix"  # Replace with your movie title
    user_id = 1           # Replace with your user ID
    recommendations = hybrid_model.get_hybrid_recommendations(title, user_id, top_n=10)

    print("Hybrid Recommendations:")
    print(recommendations)

    # Save the hybrid recommendation system model
    hybrid_model.save_model(model_save_path)
