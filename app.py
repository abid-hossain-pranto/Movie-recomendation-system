import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from pymongo import MongoClient
import bcrypt
import numpy as np
from sklearn.metrics import precision_score, f1_score, accuracy_score, confusion_matrix
import os

# ------------------- Secrets Setup -------------------
MONGO_URI = st.secrets["MONGO"]["URI"]
TMDB_API_KEY = st.secrets["TMDB"]["TMDB_API_KEY"]
TMDB_IMG_BASE_URL = st.secrets["TMDB"]["TMDB_IMG_BASE_URL"]

# ------------------- MongoDB Setup -------------------
client = MongoClient(MONGO_URI)
db = client.get_database("movie_recommendation_system")
users_collection = db.get_collection("users")

# ------------------- Poster Fetch -------------------
def get_movie_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json().get("results")
            if results:
                poster_path = results[0].get("poster_path")
                if poster_path:
                    return f"{TMDB_IMG_BASE_URL}{poster_path}"
        return "https://via.placeholder.com/300x450?text=No+Image"
    except:
        return "https://via.placeholder.com/300x450?text=No+Image"

# ------------------- Authentication -------------------
def check_user_credentials(email, password):
    user = users_collection.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return True
    return False

def register_user(email, password):
    if users_collection.find_one({"email": email}):
        return False
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({"email": email, "password": hashed_pw})
    return True

# ------------------- Load Movie Data -------------------
@st.cache_resource
def load_data():
    file_path = "movie_dict_latest.pcl"
    if not os.path.exists(file_path):
        return pd.DataFrame(), []

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        data = pd.DataFrame(data)

    genre_columns = ['action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
                     'drama', 'family', 'fantasy', 'foreign', 'history', 'horror', 'music',
                     'mystery', 'romance', 'science fiction', 'thriller', 'tv movie', 'war', 'western']
    genre_columns = [g for g in genre_columns if g in data.columns]

    data['genre_names'] = data[genre_columns].apply(lambda row: [g for g in genre_columns if row[g] == 1], axis=1)
    data['combined_features'] = data[genre_columns].astype(str).agg(" ".join, axis=1) + " " + data['overview'].astype(str)
    data['popularity'] = pd.to_numeric(data.get('popularity', 0), errors='coerce')
    data.fillna(0, inplace=True)

    return data, genre_columns

df, all_genres = load_data()
genre_options = ["All"] + sorted(all_genres)
movie_options = ["None"] + sorted(df['title'].unique().tolist())

# ------------------- Similarity -------------------
@st.cache_resource
def build_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_similarity(df)

# ------------------- Train RF Model -------------------
def train_rf_model(data, genres):
    features = ['release_year', 'popularity'] + genres
    X = data[features]
    y = data[genres]
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X, y)
    return model

rf_model = train_rf_model(df, all_genres)

# ------------------- Recommend Movies -------------------
def recommend_movies(title, data, similarity, model, genre_filter=None, n=20):
    if 'title' not in data.columns or title not in data['title'].values:
        return pd.DataFrame()

    idx = data[data['title'] == title].index[0]
    sim_scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:]]
    similar_movies = data.iloc[movie_indices]

    # Predict genres from input movie
    input_features = data.loc[idx, ['release_year', 'popularity'] + all_genres].values.reshape(1, -1)
    predicted = model.predict(input_features)[0]
    pred_genres = [all_genres[i] for i, g in enumerate(predicted) if g == 1]

    def overlap(g_list): return any(g in g_list for g in pred_genres)
    filtered = similar_movies[similar_movies['genre_names'].apply(overlap)]

    if genre_filter and genre_filter != "All":
        filtered = filtered[filtered[genre_filter] == 1]

    return filtered.head(n)

# ------------------- Metrics -------------------
def calculate_metrics(recommended_titles, ground_truth):
    y_true = np.zeros(len(all_genres))
    y_pred = np.zeros(len(all_genres))

    for g in ground_truth:
        if g in all_genres:
            y_true[all_genres.index(g)] = 1
    for t in recommended_titles:
        genres = df[df['title'] == t]['genre_names'].values
        if len(genres) > 0:
            for g in genres[0]:
                if g in all_genres:
                    y_pred[all_genres.index(g)] = 1

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred)
    }

# ------------------- Pages -------------------
def login_page():
    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_user_credentials(email, password):
            st.session_state.logged_in = True
            st.session_state.email = email
            st.session_state.page = "recommend"
        else:
            st.error("❌ Invalid credentials!")
    if st.button("No account? Register here"):
        st.session_state.page = "register"

def register_page():
    st.title("📝 Register")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if register_user(email, password):
            st.success("Account created! Please login.")
            st.session_state.page = "login"
        else:
            st.error("Email already registered.")
    if st.button("Already have an account?"):
        st.session_state.page = "login"

def recommend_page():
    st.title("🎬 Movie Recommender")
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in first!")
        st.session_state.page = "login"
        return

    st.markdown(f"Welcome, **{st.session_state.email}**!")

    movie_choice = st.selectbox("Pick a Movie:", movie_options)
    genre_choice = st.selectbox("Filter by Genre:", genre_options)

    if st.button("🎯 Recommend"):
        if movie_choice != "None":
            recs = recommend_movies(movie_choice, df, cosine_sim, rf_model, genre_choice)
        else:
            recs = df[df[genre_choice.lower()] == 1] if genre_choice != "All" else df
            recs = recs.head(20)

        if recs.empty:
            st.warning("No recommendations found.")
        else:
            st.subheader("Results:")
            cols = st.columns(5)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    st.image(get_movie_poster(row['title']), use_column_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(", ".join(row['genre_names']))

    if st.button("Logout"):
        st.session_state.clear()
        st.success("Logged out.")

# ------------------- App Routing -------------------
if 'page' not in st.session_state:
    st.session_state.page = "login"

if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "recommend":
    recommend_page()
