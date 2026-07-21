from flask import Flask, render_template, request
import pickle
import requests
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)

# ===========================
# LOAD MODEL FILES
# ===========================
movies = pickle.load(open("models/movies.pkl", "rb"))

if os.path.exists("models/similarity.pkl"):
    similarity = pickle.load(open("models/similarity.pkl", "rb"))
else:
    print("similarity.pkl not found. Recreating similarity matrix...")

    df = pd.read_csv("data/movie_features.csv")

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    print("Similarity matrix created successfully!")

# ===========================
# TMDB POSTER FUNCTION
# ===========================
def fetch_poster(movie_id):

    api_key = "45a17ec80f4b4a7282e16b2180d1141d"

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"

    response = requests.get(url)

    data = response.json()

    poster_path = data.get("poster_path")

    if poster_path:
        return "https://image.tmdb.org/t/p/w500" + poster_path

    return ""


# ===========================
# RECOMMENDATION FUNCTION
# ===========================
def recommend(movie):

    movie_index = movies[movies["title"] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:

        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)

        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


# ===========================
# HOME PAGE
# ===========================
@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    posters = []

    if request.method == "POST":

        movie = request.form["movie"]

        recommendations, posters = recommend(movie)

    movie_titles = sorted(movies["title"].values)

    return render_template(
        "index.html",
        movie_titles=movie_titles,
        recommendations=recommendations,
        posters=posters,
        zip=zip
    )


# ===========================
# RUN APP
# ===========================
if __name__ == "__main__":
    app.run(debug=True)