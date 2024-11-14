import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    raw_data_csv1 = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\ml-20m\ml-20m\genome-scores.csv'
    raw_data_csv2 = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\ml-20m\ml-20m\genome-tags.csv'
    raw_data_csv3 = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\ml-20m\ml-20m\links.csv'
    raw_data_csv4 = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\ml-20m\ml-20m\movies.csv'
    raw_data_csv5 = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\ml-20m\ml-20m\ratings.csv'
    raw_data_csv6 = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\ml-20m\ml-20m\tags.csv'
    train_data_path = os.path.join(os.getcwd(), 'artifacts', 'training_data.pkl')
    test_data_path = os.path.join(os.getcwd(), 'artifacts', 'testing_data.pkl')
    common_column = 'movieId'

class TransformationConfig:
    def __init__(self):
        self.data_config = DataConfig()
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
    
    def read_data(self):
        """Reads each raw data file and verifies its existence."""
        data_frames = {}
        paths = [
            self.data_config.raw_data_csv1,
            self.data_config.raw_data_csv2,
            self.data_config.raw_data_csv3,
            self.data_config.raw_data_csv4,
            self.data_config.raw_data_csv5,
            self.data_config.raw_data_csv6,
        ]
        
        for path in paths:
            try:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
                df = pd.read_csv(path)
                data_frames[path] = df
                logger.info(f"Loaded data from {path}")
            except Exception as e:
                logger.error(f"Error reading data from {path}: {e}", exc_info=True)
                raise
        return data_frames

    def merging_data(self):
        """Merges and optimizes data from multiple sources."""
        try:
            df_ratings = pd.read_csv(self.data_config.raw_data_csv5)
            df_links = pd.read_csv(self.data_config.raw_data_csv3)
            df_movies = pd.read_csv(self.data_config.raw_data_csv4)
            df_genome_scores = pd.read_csv(self.data_config.raw_data_csv1)
            df_tags = pd.read_csv(self.data_config.raw_data_csv6)
            
            # Initial merges
            df_merged = pd.merge(df_ratings, df_links, on=self.data_config.common_column, how='right')
            df_merged = pd.merge(df_movies, df_merged, on=self.data_config.common_column, how="right")

            # Selecting necessary columns and optimizing
            df_genome_scores_small = df_genome_scores[['movieId', 'tagId', 'relevance']]
            df_movies_small = df_merged[['movieId', 'title', 'genres', 'userId', 'rating', 'timestamp']]

            # Convert timestamps to datetime and extract additional features
            df_movies_small['datetime'] = pd.to_datetime(df_movies_small['timestamp'], unit='s')

            # Drop duplicates in the common column
            df_genome_scores_small = df_genome_scores_small.drop_duplicates(subset=[self.data_config.common_column])
            df_movies_small = df_movies_small.drop_duplicates(subset=[self.data_config.common_column])

            # Downcast numerical columns
            for col in df_genome_scores_small.select_dtypes(include=['float64', 'int64']).columns:
                df_genome_scores_small[col] = pd.to_numeric(df_genome_scores_small[col], downcast='float')
            for col in df_movies_small.select_dtypes(include=['float64', 'int64']).columns:
                df_movies_small[col] = pd.to_numeric(df_movies_small[col], downcast='float')

            # Chunked merge for memory efficiency
            chunk_size = 100000
            merged_chunks = []
            for chunk in pd.read_csv(self.data_config.raw_data_csv1, chunksize=chunk_size):
                chunk = chunk[['movieId', 'tagId', 'relevance']]
                merged_chunk = pd.merge(df_movies_small, chunk, on=self.data_config.common_column, how='inner')
                merged_chunks.append(merged_chunk)

            self.dfmerged_final = pd.concat(merged_chunks, ignore_index=True)
            logger.info("Data merged successfully.")

            # Final merge with tags data
            self.dfmerged_final = pd.merge(self.dfmerged_final, df_tags, on=['userId', 'movieId'], how='right')

            # Dropping unwanted columns
            self.dfmerged_final = self.dfmerged_final.drop(columns=['tag', 'timestamp_x', 'timestamp_y'])
            self.dfmerged_final.drop_duplicates()
            
            return self.dfmerged_final

        except Exception as e:
            logger.error(f"Error during data merging: {e}", exc_info=True)
            raise
    
    def handling_missing_value(self):
        """Handles missing values in the merged DataFrame."""
        try:
            self.dfmerged_final = self.dfmerged_final.dropna()
            logger.info("Missing values handled successfully.")
        except Exception as e:
            logger.error(f"Error in handling missing values: {e}", exc_info=True)
            raise

    def label_encode(self):
        """Applies label encoding to specified columns."""
        try:
            for col in ['title', 'genres']:
                if col in self.dfmerged_final.columns:
                    logger.info(f"Encoding column {col} using label encoding.")
                    self.dfmerged_final[col] = self.encoder.fit_transform(self.dfmerged_final[col])
            logger.info("Label encoding completed.")
        except Exception as e:
            logger.error(f"Error in label encoding: {e}", exc_info=True)
            raise

    def save_transformed_data(self, train_df, test_df):
        """Saves the training and testing data to separate files."""
        logger.info("Saving the transformed training and testing datasets in pickle format.")
        try:
            os.makedirs(os.path.dirname(self.data_config.train_data_path), exist_ok=True)
            with open(self.data_config.train_data_path, 'wb') as f:
                pickle.dump(train_df, f)
            logger.info(f"Training data saved to {self.data_config.train_data_path}.")

            with open(self.data_config.test_data_path, 'wb') as f:
                pickle.dump(test_df, f)
            logger.info(f"Testing data saved to {self.data_config.test_data_path}.")
        except Exception as e:
            logger.error(f"Error in saving transformed data: {e}", exc_info=True)
            raise

    def transform(self):
        """Executes the entire transformation pipeline."""
        try:
            logger.info("Starting data transformation process.")
            self.merging_data()
            self.label_encode()
            self.handling_missing_value()

            # Split the data into training and testing sets
            train_df, test_df = train_test_split(self.dfmerged_final, test_size=0.2, random_state=42)
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


