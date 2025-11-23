# Movie Recommender System 🎬

A content-based movie recommendation system built with Streamlit that suggests movies based on either a movie title or custom keywords. The system uses TF-IDF vectorization and cosine similarity to find movies with similar content.

## Features

**Movie Title Search**: Enter a movie title to get 10 similar movie recommendations

**Keyword Search**: Enter custom keywords to discover movies matching those themes

**Rich Information**: View movie overviews, ratings, and release dates for each recommendation

**Interactive UI**: Clean and user-friendly Streamlit interface

## How It Works

The recommender system uses TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize movie metadata (genres, keywords, cast, crew, etc.), Cosine Similarity to measure the similarity between movies based on their content, and Content-Based Filtering to recommend movies with similar characteristics.

## Installation

Clone this repository:
```bash
git clone https://github.com/moustt22/Movie-Recommender.git
cd Movie-Recommender
```

Install required dependencies:
```bash
pip install streamlit pandas scikit-learn joblib
```

Ensure you have the required data files in the project directory: `movies_df_final.pkl` (cleaned movie dataframe), `tfidf_matrix_final.pkl` (pre-computed TF-IDF matrix), and `tfidf_vectorizer.pkl` (fitted TF-IDF vectorizer).

## Usage

Run the Streamlit app:
```bash
streamlit run movie_rec.py
```

The app will open in your browser. You can then choose between "Movie Title" or "Keywords" input mode, enter your search query, and click the recommend button to see results.

### Example Queries

By Movie Title: "The Dark Knight", "Inception", "Toy Story"

By Keywords: "space adventure aliens", "romantic comedy love", "superhero action"

## Project Structure

```
Movie-Recommender/
│
├── movie_rec.py              # Main Streamlit application
├── movies_df_final.pkl       # Preprocessed movie dataset
├── tfidf_matrix_final.pkl    # TF-IDF matrix of movie features
├── tfidf_vectorizer.pkl      # Trained TF-IDF vectorizer
└── README.md                 # Project documentation
```

## Requirements

Python 3.7+, streamlit, pandas, scikit-learn, joblib

## Data

The system requires pre-processed movie data including movie titles, overviews/descriptions, vote averages (ratings), release dates, and combined keyword/genre/cast/crew information (used for similarity).

## Future Improvements

Add movie posters and images, implement collaborative filtering, add user ratings and feedback, include more filtering options (genre, year, rating), and deploy to cloud platform (Streamlit Cloud, Heroku, etc.).


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Movie data sourced from [TMDB]([https://www.themoviedb.org/](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv)), built with [Streamlit](https://streamlit.io/), powered by [scikit-learn](https://scikit-learn.org/).
