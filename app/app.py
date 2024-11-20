from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
import tensorflow as tf
import os
import requests
import sys

# Add project root to Python path
sys.path.append("..")

from model import BERT4Rec 

# Flask app initialization
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", 'importfromthemostaccuratelyusedmodel')
bcrypt = Bcrypt(app)

# MongoDB connection
MONGO_URI = os.getenv(
    'MONGO_URI',
    'mongodb+srv://davetsegaye526:0985165082@cluster0.wzvj9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
)
client = MongoClient(MONGO_URI)
db = client['movie_recommendation_db']
users_collection = db['users']
movies_collection = db['movies']

# TMDb API configuration
TMDB_API_KEY = os.getenv(
    "TMDB_API_KEY",
    "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyZGM0ZjhkMTY4NjQ1ZGE0NjUzODcwZmQ3NmFhMWQ1YyIsIm5iZiI6MTczMTk5OTE4My4xMDkxNjA3LCJzdWIiOiI2NzNjMzIwOTA2ZmQ4ODVhYjlkZDhmMTgiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.czpwVSJTCQ6Jc6_DVilpMZ_aSIdQ1EJsoWDpWHMhMmA"
)
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Define model path and load the model
MODEL_PATH = '../models/bert4rec_model.keras'
try:
    recommendation_model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'BERT4Rec': BERT4Rec})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


# Routes

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = users_collection.find_one({'username': username})
        
        if existing_user:
            return render_template('register.html', error="Username already exists.")
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        users_collection.insert_one({'username': username, 'password': hashed_password})
        return redirect(url_for('login_user'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username})
        
        if user and bcrypt.check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login_user'))
    
    if request.method == 'POST':
        return redirect(url_for('recommend_movies'))
    
    return render_template('dashboard.html', username=session['username'])


@app.route('/recommend', methods=['POST'])
def recommend_movies():
    if 'username' not in session:
        return redirect(url_for('login_user'))

    user_input = request.form.get('movie_title', '').strip()
    if not user_input:
        return render_template('dashboard.html', error="Please provide a movie title.")

    # Retrieve movie details from MongoDB
    movie = movies_collection.find_one({'title': {'$regex': f"^{user_input}$", '$options': 'i'}})
    if not movie:
        return render_template('dashboard.html', error="Movie not found.")

    # Predict similar movies using the ML model
    movie_id = movie['movieId']
    try:
        recommended_ids = recommendation_model.predict([[movie_id]])
        recommended_ids = recommended_ids.flatten().tolist()
    except Exception as e:
        return render_template('dashboard.html', error=f"Error during recommendation: {str(e)}")

    # Fetch recommended movies from the database
    recommended_movies = list(movies_collection.find(
        {'movieId': {'$in': recommended_ids}},
        {'_id': 0, 'title': 1}
    ))

    # Fetch movie details from TMDb API
    recommendations = []
    for movie in recommended_movies:
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": movie['title']}
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                movie_data = data['results'][0]
                poster_path = movie_data.get('poster_path')
                recommendations.append({
                    "title": movie_data['title'],
                    "overview": movie_data.get('overview', 'No overview available.'),
                    "release_date": movie_data.get('release_date', 'N/A'),
                    "poster_url": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
                })

    return render_template('recommend.html', recommendations=recommendations, input_movie=user_input)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
