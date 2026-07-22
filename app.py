from flask import Flask, render_template, request
import pickle
import requests
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

app = Flask(__name__)

# ===========================
# LOAD MODEL FILES
# ===========================
movies = pickle.load(open("models/movies.pkl", "rb"))
movies_data = pd.read_csv("data/tmdb_5000_movies.csv")

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

    selected_movie_id = movies.iloc[movie_index].movie_id
    selected_movie_poster = fetch_poster(selected_movie_id)

    recommended_movies = []
    recommended_posters = []
    recommended_details = []

    for i in movie_list:

        movie_id = movies.iloc[i[0]].movie_id
        movie_title = movies.iloc[i[0]].title

        recommended_movies.append(movie_title)
        recommended_posters.append(fetch_poster(movie_id))

        movie_info = movies_data[movies_data["title"] == movie_title]

        if not movie_info.empty:

            movie_info = movie_info.iloc[0]

            genres = ast.literal_eval(movie_info["genres"])
            genre_names = " • ".join([g["name"] for g in genres])

        recommended_details.append({
        "rating": f'{round(movie_info["vote_average"], 1)}/10',
        "year": str(movie_info["release_date"])[:4],
        "genre": genre_names
        })

    else:

        recommended_details.append({
                "rating": "N/A",
                "year": "N/A",
                "genre": "N/A"
            })

    return (
        selected_movie_poster,
        recommended_movies,
        recommended_posters,
        recommended_details
    )


# ===========================
# HOME PAGE
# ===========================
@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    posters = []
    details = []
    selected_poster = ""

    if request.method == "POST":

        movie = request.form["movie"]

        selected_poster, recommendations, posters, details = recommend(movie)

    movie_titles = sorted(movies["title"].values)

    return render_template(
        "index.html",
        movie_titles=movie_titles,
        recommendations=recommendations,
        posters=posters,
        details=details,
        selected_poster=selected_poster,
        zip=zip
    )


# ===========================
# RUN APP
# ===========================
if __name__ == "__main__":
    app.run(debug=True)