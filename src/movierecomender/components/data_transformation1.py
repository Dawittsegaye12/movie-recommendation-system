import os
import pickle
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    training_data = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\artifacts\training_data.pkl'
    testing_data = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\artifacts\testing_data.pkl'
    handled_training_path = os.path.join(os.getcwd(), 'artifact', 'training_data1.pkl')
    handled_testing_path = os.path.join(os.getcwd(), 'artifact', 'testing_data1.pkl')


class TransformationConfig:
    def __init__(self):
        self.config = Config()  # Instantiate the config class
        self.df = None  # Placeholder for the dataframe

    def read_data(self):
        """Reads and loads the training and testing data."""
        data_frames = {}
        paths = [
            self.config.training_data,
            self.config.testing_data
        ]
        for path in paths:
            try:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
                df = pd.read_pickle(path)
                data_frames[path] = df
                logger.info(f"Loaded data from {path}")
            except Exception as e:
                logger.error(f"Error reading data from {path}: {e}", exc_info=True)
                raise
        return data_frames

    def handling_duplicated(self):
        """Handles duplicates in the data."""
        try:
            if self.df is not None:
                self.df = self.df.drop_duplicates()
                logger.info("Duplicates removed successfully.")
            else:
                logger.warning("Dataframe is not loaded yet.")
        except Exception as e:
            logger.error(f"Error in handling duplicates: {e}", exc_info=True)
            raise

    def save_transformed_data(self, train_df, test_df):
        """Saves the transformed training and testing datasets."""
        logger.info("Saving the transformed datasets in pickle format.")
        try:
            os.makedirs(os.path.dirname(self.config.handled_training_path), exist_ok=True)
            with open(self.config.handled_training_path, 'wb') as f:
                pickle.dump(train_df, f)
            logger.info(f"Training data saved to {self.config.handled_training_path}.")

            with open(self.config.handled_testing_path, 'wb') as f:
                pickle.dump(test_df, f)
            logger.info(f"Testing data saved to {self.config.handled_testing_path}.")
        except Exception as e:
            logger.error(f"Error in saving transformed data: {e}", exc_info=True)
            raise

    def transform(self):
        """Executes the entire transformation pipeline."""
        try:
            logger.info("Starting data transformation process.")
            # Read data
            data_frames = self.read_data()

            # Assume we are working with the training data for simplicity
            self.df = data_frames[self.config.training_data]

            # Handle duplicates
            self.handling_duplicated()

            # Sequence creation logic (movieId, genres, datetime, title)
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df = self.df.drop(columns=['rating', 'tagId', 'relevance'])  # Drop unnecessary columns
            self.df = self.df.sort_values(by=['userId', 'datetime'])  # Sort by userId and datetime

            user_sequences_with_details = self.df.groupby('userId').agg(
                movie_sequence=('movieId', lambda x: list(x)),
                genre_sequence=('genres', lambda x: list(x)),
                datetime_sequence=('datetime', lambda x: list(x)),
                title_sequence=('title', lambda x: list(x))
            ).reset_index()

            # Merge the sequences back to the original dataframe
            transformed_df = pd.merge(self.df, user_sequences_with_details[['userId', 'movie_sequence', 'genre_sequence', 'datetime_sequence', 'title_sequence']], on='userId', how='left')

            # Split the data into training and testing sets
            train_df, test_df = train_test_split(transformed_df, test_size=0.2, random_state=42)
            logger.info("Data split into training and testing sets.")

            # Save the transformed data
            self.save_transformed_data(train_df, test_df)
            logger.info("Data transformation process completed successfully.")
        except Exception as e:
            logger.error(f"Error in the data transformation process: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    transformer = TransformationConfig()
    transformer.transform()
