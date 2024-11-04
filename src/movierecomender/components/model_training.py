import os
import sys
import pandas as pd
import pickle
import logging
from difflib import get_close_matches

# Ensure the correct path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.movierecomender.components.collaborative import CollaborativeFiltering
from src.movierecomender.components.content_based import ContentBasedFiltering

logging.basicConfig(level=logging.INFO)

class HybridRecommendationSystem:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.user_movie_matrix = None
        self.collaborative_filtering = None
        self.content_based_filtering = None

    def load_data(self, train_data_path, test_data_path):
        logging.info("Loading data...")
        try:
            with open(train_data_path, 'rb') as f:
                self.train_data = pickle.load(f)
            logging.info("Training data loaded successfully.")
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            logging.error(f"Error loading pickle file: {e}. Attempting to load from CSV.")
            try:
                self.train_data = pd.read_csv(train_data_path)
                logging.info("Training data loaded from CSV successfully.")
            except FileNotFoundError:
                logging.critical(f"Failed to load training data from CSV format: {e}")
                self.train_data = None

        try:
            with open(test_data_path, 'rb') as f:
                self.test_data = pickle.load(f)
            logging.info("Testing data loaded successfully.")
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            logging.error(f"Error loading pickle file: {e}. Attempting to load from CSV.")
            try:
                self.test_data = pd.read_csv(test_data_path)
                logging.info("Testing data loaded from CSV successfully.")
            except FileNotFoundError:
                logging.critical(f"Failed to load testing data from CSV format: {e}")
                self.test_data = None

        if self.train_data is None or self.test_data is None:
            logging.critical("Failed to load both training and testing data. Exiting.")
            sys.exit(1)

    def preprocess_data(self):
        logging.info("Preprocessing data to define dependent and independent variables...")

        # Defining 'title' as the dependent variable
        self.train_data['title'] = self.train_data['title'].astype(str)
        independent_vars = self.train_data.drop(columns=['title'])
        dependent_var = self.train_data['title']

        logging.info("Creating User-Item Matrix based on ratings...")
        self.user_movie_matrix = independent_vars.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
        logging.info("User-Item Matrix created successfully with NaN handling.")

        # Initialize collaborative and content-based filtering
        self.collaborative_filtering = CollaborativeFiltering(self.user_movie_matrix)
        self.content_based_filtering = ContentBasedFiltering(self.train_data)
        self.content_based_filtering.preprocess()
        logging.info("Content-based movie profiles created successfully.")

    def get_hybrid_recommendations(self, title, user_id, top_n=10):
        logging.info(f"Generating hybrid recommendations for title '{title}' and user '{user_id}'...")

        # Convert titles to strings to avoid TypeErrors with get_close_matches
        titles = list(map(str, self.train_data['title'].unique()))

        if title not in titles:
            close_matches = get_close_matches(title, titles, n=5, cutoff=0.5)
            logging.error(f"Movie title '{title}' not found. Did you mean one of these? {close_matches}")
            return pd.DataFrame()  # Return an empty DataFrame if title not found

        content_recommendations = self.content_based_filtering.get_recommendations(title, top_n)
        collaborative_recommendations = self.collaborative_filtering.get_recommendations(user_id, top_n)

        combined_recommendations = pd.concat([content_recommendations, collaborative_recommendations])
        hybrid_recommendations = combined_recommendations.groupby('title').mean().sort_values('rating', ascending=False)
        logging.info("Hybrid recommendations generated successfully.")
        return hybrid_recommendations.head(top_n)

    def save_model(self, file_path=None):
        if file_path is None:
            directory = os.path.join(os.getcwd(), 'models')
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, 'hybrid_recommendation_model.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"Model saved to {file_path}")

if __name__ == '__main__':
    train_data_path = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\artifacts\training_data.pkl'
    test_data_path = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\artifacts\testing_data.pkl'

    hybrid_model = HybridRecommendationSystem()
    hybrid_model.load_data(train_data_path, test_data_path)
    hybrid_model.preprocess_data()

    title = "Toy Story"
    user_id = 1
    recommendations = hybrid_model.get_hybrid_recommendations(title, user_id, top_n=10)
    print("Hybrid Recommendations:")
    print(recommendations)

    hybrid_model.save_model()
