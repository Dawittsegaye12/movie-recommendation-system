import pandas as pd
import logging

class ContentBasedFiltering:
    def __init__(self, train_data):
        self.train_data = train_data
        self.movie_profiles = None

    def preprocess(self):
        logging.info("Preprocessing data to create content-based movie profiles...")

        # Convert 'rating' to numeric, coercing errors to NaN
        self.train_data['rating'] = pd.to_numeric(self.train_data['rating'], errors='coerce')

        # Drop rows where 'rating' is NaN after conversion
        self.train_data = self.train_data.dropna(subset=['rating'])

        # Group by 'movieId' and calculate the mean rating for each movie
        self.movie_profiles = self.train_data.groupby('movieId')['rating'].mean()

        logging.info("Content-based movie profiles created successfully.")

    def get_movie_profiles(self):
        return self.movie_profiles
