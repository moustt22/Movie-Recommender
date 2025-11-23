import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# -------------------------
# Load saved files
# -------------------------
df = joblib.load('movies_df_final.pkl')  # your cleaned movie dataframe
tfidf_matrix = joblib.load('tfidf_matrix_final.pkl')  # TF-IDF matrix of keywords_str
tfidf = joblib.load('tfidf_vectorizer.pkl')  # TF-IDF vectorizer

# -------------------------
# Build movie title → index mapping
# -------------------------
movie_index=pd.Series(df.index,index=df['title'].str.lower())

# -------------------------
# Recommendation function
# -------------------------
def recommend_by_title(movie_title):
    movie_title = movie_title.lower()

    if movie_title not in movie_index:
        return []  # Movie not found

    idx = movie_index[movie_title]
    query = tfidf_matrix[idx]  # single movie vector
    sim = cosine_similarity(query, tfidf_matrix).flatten()  # 1D array
    rec_idx = (-sim).argsort()[1:11]  # top N excluding the movie itself

    return rec_idx

def recommend_by_keywords(s):
    new_s=tfidf.transform([s])
    score=cosine_similarity(new_s, tfidf_matrix).flatten()
    rec_idx=(-score).argsort()[1:11]
    return rec_idx



# -------------------------
# Streamlit UI
# -------------------------
st.title("🎬 Movie Recommender System")


mode = st.radio("Choose input type:", ["Movie Title", "Keywords"])

if mode == "Movie Title":
    movie_name = st.text_input("Enter a movie title:")
    if st.button("Recommend by Title"):
        results = recommend_by_title(movie_name)
        if len(results):
            st.write("### Recommended Movies:")
            for i in results:
                st.write(f"**Title:** {df['title'].iloc[i]}")
                st.write(f"**Overview:** {df['overview'].iloc[i]}")
                st.write(f"**Rating:** {df['vote_average'].iloc[i]}")
                st.write(f"**Release Date:** {df['release_date'].iloc[i]}")
                st.write("---")
        else:
            st.write("❌ Movie not found!")

else:
    keywords = st.text_input("Enter keywords (separated by spaces):")
    if st.button("Recommend by Keywords"):
        results = recommend_by_keywords(keywords)
        if len(results):
            st.write("### Recommended Movies:")
            for i in results:
                st.write(f"**Title:** {df['title'].iloc[i]}")
                st.write(f"**Overview:** {df['overview'].iloc[i]}")
                st.write(f"**Rating:** {df['vote_average'].iloc[i]}")
                st.write(f"**Release Date:** {df['release_date'].iloc[i]}")
                st.write("---")
        else:
            st.write("❌ No movies match those keywords!")