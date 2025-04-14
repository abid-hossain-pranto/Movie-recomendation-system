import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
from pymongo import MongoClient
import bcrypt
from sklearn.ensemble import RandomForestClassifier

# ---------------- MongoDB Setup ----------------
MONGO_URI = st.secrets["MONGO"]["URI"]
client = MongoClient(MONGO_URI)
db = client.get_database("movie_recommendation_system")
users_collection = db.get_collection("users")

# ---------------- TMDb API Setup ----------------
TMDB_API_KEY = st.secrets["TMDB"]["TMDB_API_KEY"]
TMDB_IMG_BASE_URL = st.secrets["TMDB"]["TMDB_IMG_BASE_URL"]

def get_movie_poster(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("results")
        if results:
            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"{TMDB_IMG_BASE_URL}{poster_path}"
    return "https://via.placeholder.com/300x450?text=No+Image"

# ---------------- MongoDB Authentication ----------------
def check_user_credentials(email, password):
    user = users_collection.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return True
    return False

def register_user(email, password):
    if users_collection.find_one({"email": email}):
        return False  # Email already registered
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({"email": email, "password": hashed_password})
    return True

# ---------------- Load Movie Data ----------------
@st.cache_resource
def load_data():
    with open('movie_dict_latest.pcl', 'rb') as file:
        data = pickle.load(file)
    if isinstance(data, dict):
        data = pd.DataFrame(data)

    data['popularity'] = pd.to_numeric(data.get('popularity', 0), errors='coerce')
    data.fillna(0, inplace=True)

    genre_columns = ['action', 'adventure', 'animation', 'comedy', 'crime', 'documentary', 
                     'drama', 'family', 'fantasy', 'foreign', 'history', 'horror', 'music', 
                     'mystery', 'romance', 'science fiction', 'thriller', 'tv movie', 'war', 'western']
    genre_columns = [col for col in genre_columns if col in data.columns]

    data['genre_names'] = data[genre_columns].apply(
        lambda row: [col for col in genre_columns if row.get(col, 0) == 1], axis=1
    )
    
    data['combined_features'] = data[genre_columns].astype(str).agg(' '.join, axis=1) + " " + data['overview'].astype(str)

    return data, genre_columns

df, all_genres = load_data()
genre_options = ["All"] + sorted(all_genres)
movie_options = ["None"] + sorted(df['title'].unique().tolist())

# ---------------- Build TF-IDF and Similarity Matrix ----------------
@st.cache_resource
def build_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_similarity(df)

# ---------------- Train Random Forest Model ----------------
def train_rf_model(data, genre_columns):
    data['combined_features'] = data[genre_columns].astype(str).agg(' '.join, axis=1) + " " + data['overview'].astype(str)

    # Select Features & Labels for Classification Model
    features = ['release_year', 'popularity'] + genre_columns
    X = data[features]
    y = data[genre_columns]

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
        warm_start=True
    )

    rf.fit(X, y)
    return rf

# Train the RandomForest model
rf_model = train_rf_model(df, all_genres)

# ---------------- Recommendation Logic ----------------
def recommend_movies(movie_title, data, similarity_matrix, rf_model, genre_filter=None, n_recommendations=20):
    if 'title' not in data.columns or movie_title not in data['title'].values:
        return pd.DataFrame()

    idx = data[data['title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:]]

    similar_movies = data.iloc[movie_indices]

    # Predict genres using Random Forest
    input_features = data.loc[idx, ['release_year', 'popularity'] + all_genres].values.reshape(1, -1)
    predicted_genres = rf_model.predict(input_features)[0]
    predicted_genre_names = [all_genres[i] for i, val in enumerate(predicted_genres) if val == 1]

    # Filter movies by predicted genre overlap
    def genre_overlap(genres):
        return any(g in genres for g in predicted_genre_names)

    filtered_movies = similar_movies[similar_movies['genre_names'].apply(genre_overlap)]

    if genre_filter and genre_filter != "All":
        filtered_movies = filtered_movies[filtered_movies[genre_filter] == 1]

    return filtered_movies.head(n_recommendations)

# ---------------- Evaluation Metrics ----------------
def calculate_metrics(recommended_titles, ground_truth_genres):
    y_true = np.zeros(len(all_genres))
    y_pred = np.zeros(len(all_genres))

    for genre in ground_truth_genres:
        if genre in all_genres:
            y_true[all_genres.index(genre)] = 1

    for title in recommended_titles:
        genres = df[df['title'] == title]['genre_names'].values
        if genres.any():
            for genre in genres[0]:
                if genre in all_genres:
                    y_pred[all_genres.index(genre)] = 1

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, conf_matrix

# ---------------- Page Routing ----------------
def login_page():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_user_credentials(email, password):
            st.session_state["logged_in"] = True
            st.session_state["email"] = email
            st.session_state.page = "recommendations"
        else:
            st.error("Invalid email or password!")

    if st.button("Don't have an account? Register here"):
        st.session_state.page = "register"

def register_page():
    st.title("Register")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type="password")

    if st.button("Register"):
        if not register_user(new_email, new_password):
            st.error("Email already registered!")
        else:
            st.success("Registration successful! Please login.")
            st.session_state.page = "login"

    if st.button("Already have an account? Login here"):
        st.session_state.page = "login"

def recommendation_page():
    st.title("Movie Recommendation")
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("Please log in to see recommendations!")
        st.session_state.page = "login"
        return

    st.write(f"Hello, {st.session_state['email']}! You're logged in.")

    selected_movie = st.selectbox("Choose a Movie (Optional):", movie_options)
    selected_genre = st.selectbox("Filter by Genre (Optional):", genre_options)

    if st.button("🎯 Recommend"):
        if selected_movie != "None":
            recs = recommend_movies(selected_movie, df, cosine_sim, rf_model, selected_genre if selected_genre != "All" else None)
        else:
            recs = df.copy()
            if selected_genre != "All":
                recs = recs[recs[selected_genre.lower()] == 1]
            recs = recs.head(20)

        if recs.empty:
            st.warning("😕 No recommendations found with the given filters.")
        else:
            st.subheader("🎬 You might enjoy:")
            num_cols = 5
            rows = (len(recs) + num_cols - 1) // num_cols

            for row_idx in range(rows):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    rec_idx = row_idx * num_cols + col_idx
                    if rec_idx < len(recs):
                        movie_title = recs.iloc[rec_idx]['title']
                        poster_url = get_movie_poster(movie_title)
                        overview = recs.iloc[rec_idx]['overview']
                        genre_list = recs.iloc[rec_idx]['genre_names']

                        with cols[col_idx]:
                            st.image(poster_url, caption=movie_title, use_container_width=True)
                            st.markdown(f"**Genre:** {', '.join(genre_list)}")
                            with st.expander("Show Overview"):
                                st.markdown(f"**Overview:** {overview}")

    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state.page = "login"

# ---------------- Main Logic ----------------
if 'page' not in st.session_state:
    st.session_state.page = "login"

if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "recommendations":
    recommendation_page()

st.markdown("---")
st.caption("Created by MD Abid")
st.caption("🚀 *Powered by TMDb API, TF-IDF & Random forest with cosine similarity in Streamlit*")
