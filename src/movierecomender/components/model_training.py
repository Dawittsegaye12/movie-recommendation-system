import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Move the BERT4Rec class outside of the build_model method
class BERT4Rec(tf.keras.Model):
    def __init__(self, num_movies, embedding_dim, sequence_length, num_heads):
        super(BERT4Rec, self).__init__()
        self.movie_embedding = layers.Embedding(input_dim=num_movies, output_dim=embedding_dim)
        self.position_embedding = layers.Embedding(input_dim=sequence_length, output_dim=embedding_dim)
        self.transformer_block = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.output_layer = layers.Dense(num_movies, activation='softmax')

    def call(self, inputs):
        movie_sequence, position_sequence = inputs
        movie_emb = self.movie_embedding(movie_sequence)
        position_emb = self.position_embedding(position_sequence)
        x = movie_emb + position_emb
        x = self.transformer_block(x, x)
        x = x[:, -1, :]
        output = self.output_layer(x)
        return output

class BERT4RecModel:
    def __init__(self, embedding_dim=32, sequence_length=50, num_heads=4, learning_rate=1e-4):
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.model = None
        self.movie_id_map = {}

    def load_data(self, train_data_path, test_data_path):
        try:
            logger.info("Loading training and testing data from pickle files...")
            train_df = pd.read_pickle(train_data_path)
            test_df = pd.read_pickle(test_data_path)
            return train_df, test_df
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)

    def preprocess_data(self, train_df, test_df):
        logger.info("Preprocessing data...")
        
        # Convert 'datetime' to datetime type and sort by userId and datetime
        train_df['datetime'] = pd.to_datetime(train_df['datetime'])
        test_df['datetime'] = pd.to_datetime(test_df['datetime'])
        train_df = train_df.sort_values(by=['userId', 'datetime'])
        test_df = test_df.sort_values(by=['userId', 'datetime'])

        # Create a mapping for movie IDs
        unique_movie_ids = np.unique(np.concatenate([train_df['movieId'].values, test_df['movieId'].values]))
        self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
        
        # Create sequences
        train_sequences = self.create_sequences(train_df)
        test_sequences = self.create_sequences(test_df)
        
        # Prepare data for model
        train_movie_seq, train_position_seq, train_next_movie = self.prepare_data(train_sequences)
        test_movie_seq, test_position_seq, test_next_movie = self.prepare_data(test_sequences)
        
        return (train_movie_seq, train_position_seq, train_next_movie), (test_movie_seq, test_position_seq, test_next_movie)

    def create_sequences(self, df):
        user_sequences = []
        user_ids = df['userId'].unique()
        
        for user_id in user_ids:
            user_data = df[df['userId'] == user_id]
            movie_sequence = [self.movie_id_map[movie_id] for movie_id in user_data['movieId'].values]
            
            for i in range(len(movie_sequence) - self.sequence_length):
                user_sequences.append((movie_sequence[i:i + self.sequence_length], 
                                       list(range(self.sequence_length))))


        return user_sequences

    def prepare_data(self, sequences):
        movie_sequences = [seq[0] for seq in sequences]
        position_sequences = [seq[1] for seq in sequences]
        next_movie = [seq[0][-1] for seq in sequences]
        
        # Pad sequences
        movie_sequences_padded = pad_sequences(movie_sequences, padding='post', maxlen=self.sequence_length)
        position_sequences_padded = pad_sequences(position_sequences, padding='post', maxlen=self.sequence_length)
        
        return np.array(movie_sequences_padded), np.array(position_sequences_padded), np.array(next_movie)

    def build_model(self, num_movies):
        self.model = BERT4Rec(num_movies, self.embedding_dim, self.sequence_length, self.num_heads)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                           loss='sparse_categorical_crossentropy')
        logger.info("Model built successfully.")

    def train_model(self, train_data, epochs=5):
        train_movie_seq, train_position_seq, train_next_movie = train_data
        self.model.fit([train_movie_seq, train_position_seq], train_next_movie, epochs=epochs)
        logger.info("Model training complete.")

    def evaluate_model(self, test_data):
        test_movie_seq, test_position_seq, test_next_movie = test_data
        loss = self.model.evaluate([test_movie_seq, test_position_seq], test_next_movie)
        logger.info(f"Model evaluation complete. Loss: {loss}")

    def save_model(self, file_path):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)
        logger.info(f"Model saved to {file_path}")

# Main code to train, evaluate, and save the model
if __name__ == '__main__':
    train_data_path = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\artifact\training_data1.pkl'
    test_data_path = r'C:\Users\SOOQ ELASER\movie_recomendation_collaborative_filtering\artifact\training_data1.pkl'
    
    bert4rec = BERT4RecModel()
    train_df, test_df = bert4rec.load_data(train_data_path, test_data_path)
    train_data, test_data = bert4rec.preprocess_data(train_df, test_df)
    
    num_movies = len(bert4rec.movie_id_map)
    bert4rec.build_model(num_movies)
    
    bert4rec.train_model(train_data, epochs=5)
    bert4rec.evaluate_model(test_data)
    
    model_save_path = os.path.join(os.getcwd(), 'models', 'bert4rec_model.pkl')
    bert4rec.save_model(model_save_path)
