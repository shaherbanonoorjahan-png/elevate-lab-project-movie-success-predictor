# ============================================================
# MOVIE SUCCESS PREDICTION AND SENTIMENT STUDY
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Download VADER lexicon
nltk.download("vader_lexicon")


# ============================================================
# 2. LOAD MOVIE DATA
# ============================================================

# ------------------------------------------------------------
# OPTION A:
# Use your own CSV file
#
# Example:
# df = pd.read_csv("movies.csv")
#
# ------------------------------------------------------------

# Sample dataset is created here so the program can run
# immediately without needing a CSV file.

data = {
    "Title": [
        "Avengers",
        "Titanic",
        "Avatar",
        "Joker",
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Frozen",
        "Toy Story",
        "Black Panther",
        "Spider-Man",
        "Iron Man",
        "The Lion King",
        "Parasite",
        "Dune"
    ],

    "Genre": [
        "Action",
        "Romance",
        "Action",
        "Drama",
        "Sci-Fi",
        "Sci-Fi",
        "Action",
        "Animation",
        "Animation",
        "Action",
        "Action",
        "Action",
        "Animation",
        "Drama",
        "Sci-Fi"
    ],

    "Rating": [
        8.0, 7.9, 7.8, 8.4, 8.8,
        8.7, 9.0, 7.5, 8.3, 7.3,
        7.4, 7.9, 6.8, 8.5, 8.0
    ],

    "Votes": [
        1500000, 1300000, 1400000, 1500000, 2000000,
        2200000, 2800000, 700000, 1000000, 800000,
        900000, 1100000, 700000, 900000, 600000
    ],

    "Budget": [
        250, 200, 237, 55, 160,
        165, 185, 150, 30, 200,
        175, 140, 260, 11, 190
    ],

    "BoxOffice": [
        1500, 2200, 2900, 1070, 839,
        730, 1005, 1450, 373, 1340,
        825, 585, 1600, 260, 410
    ],

    "Review": [
        "Amazing movie with fantastic action and great characters.",
        "A beautiful and emotional love story. Absolutely wonderful.",
        "Excellent visuals and an amazing cinematic experience.",
        "A dark and powerful performance. Very impressive movie.",
        "Brilliant story with excellent direction and mind blowing visuals.",
        "Amazing science fiction movie with emotional storytelling.",
        "One of the best superhero movies ever made. Fantastic!",
        "Beautiful animation and a wonderful family story.",
        "A fun and emotional movie. Great characters.",
        "Good action but the story was sometimes weak.",
        "Very entertaining and exciting superhero movie.",
        "Excellent action and a great superhero character.",
        "Fun animation but the story could have been better.",
        "Brilliant story with excellent acting and direction.",
        "Amazing visuals and an interesting science fiction story."
    ]
}

df = pd.DataFrame(data)


# ============================================================
# 3. DISPLAY DATA
# ============================================================

print("\n================ MOVIE DATA ================\n")

print(df)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 4. BASIC STATISTICS
# ============================================================

print("\n================ BASIC STATISTICS ================\n")

print(df.describe())


# ============================================================
# 5. VADER SENTIMENT ANALYSIS
# ============================================================

print("\n================ SENTIMENT ANALYSIS ================\n")

sid = SentimentIntensityAnalyzer()


def get_sentiment(review):

    score = sid.polarity_scores(review)

    compound = score["compound"]

    if compound >= 0.05:
        sentiment = "Positive"

    elif compound <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return sentiment


# Calculate sentiment scores
df["Sentiment"] = df["Review"].apply(get_sentiment)

df["Sentiment_Score"] = df["Review"].apply(
    lambda x: sid.polarity_scores(x)["compound"]
)


print(df[[
    "Title",
    "Review",
    "Sentiment",
    "Sentiment_Score"
]])


# ============================================================
# 6. SENTIMENT COUNT
# ============================================================

print("\n================ SENTIMENT COUNTS ================\n")

sentiment_count = df["Sentiment"].value_counts()

print(sentiment_count)


# ============================================================
# 7. SENTIMENT VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Sentiment"
)

plt.title("Movie Review Sentiment Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.show()


# ============================================================
# 8. RATING DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Rating"],
    bins=6
)

plt.title("IMDb Movie Rating Distribution")

plt.xlabel("IMDb Rating")

plt.ylabel("Number of Movies")

plt.tight_layout()

plt.show()


# ============================================================
# 9. BOX OFFICE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["BoxOffice"],
    bins=7
)

plt.title("Movie Box Office Distribution")

plt.xlabel("Box Office Revenue")

plt.ylabel("Number of Movies")

plt.tight_layout()

plt.show()


# ============================================================
# 10. RATING VS BOX OFFICE
# ============================================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Rating",
    y="BoxOffice",
    hue="Genre",
    s=100
)

plt.title("IMDb Rating vs Box Office")

plt.xlabel("IMDb Rating")

plt.ylabel("Box Office")

plt.tight_layout()

plt.show()


# ============================================================
# 11. GENRE-WISE SENTIMENT ANALYSIS
# ============================================================

genre_sentiment = df.groupby(
    "Genre"
)["Sentiment_Score"].mean().sort_values()


print("\n================ GENRE-WISE SENTIMENT ================\n")

print(genre_sentiment)


# ============================================================
# 12. GENRE-WISE SENTIMENT VISUALIZATION
# ============================================================

plt.figure(figsize=(9, 5))

genre_sentiment.plot(
    kind="bar"
)

plt.title("Average Sentiment Score by Genre")

plt.xlabel("Genre")

plt.ylabel("Average Sentiment Score")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 13. GENRE-WISE BOX OFFICE
# ============================================================

genre_boxoffice = df.groupby(
    "Genre"
)["BoxOffice"].mean().sort_values(ascending=False)


print("\n================ GENRE-WISE BOX OFFICE ================\n")

print(genre_boxoffice)


plt.figure(figsize=(9, 5))

genre_boxoffice.plot(
    kind="bar"
)

plt.title("Average Box Office by Genre")

plt.xlabel("Genre")

plt.ylabel("Average Box Office")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ============================================================
# 14. CORRELATION ANALYSIS
# ============================================================

print("\n================ CORRELATION ANALYSIS ================\n")

numeric_columns = [
    "Rating",
    "Votes",
    "Budget",
    "BoxOffice",
    "Sentiment_Score"
]

correlation = df[numeric_columns].corr()

print(correlation)


# Correlation heatmap

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Movie Data Correlation")

plt.tight_layout()

plt.show()


# ============================================================
# 15. SUCCESS LABEL
# ============================================================

# Define movie success based on box office.
#
# Movies with BoxOffice >= 800 are considered successful.
#
# This threshold can be changed according to your dataset.

df["Success"] = np.where(
    df["BoxOffice"] >= 800,
    1,
    0
)

print("\n================ MOVIE SUCCESS ================\n")

print(
    df[[
        "Title",
        "BoxOffice",
        "Success"
    ]]
)


# ============================================================
# 16. REGRESSION MODEL
# ============================================================

# We will predict BoxOffice using:
#
# Rating
# Votes
# Budget
# Sentiment Score

features = [
    "Rating",
    "Votes",
    "Budget",
    "Sentiment_Score"
]

X = df[features]

y = df["BoxOffice"]


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Create model

model = LinearRegression()


# Train model

model.fit(
    X_train,
    y_train
)


# Predict

y_pred = model.predict(X_test)


# ============================================================
# 17. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n================ MODEL RESULTS ================\n")

print("Mean Absolute Error :", round(mae, 2))

print("Mean Squared Error  :", round(mse, 2))

print("Root Mean Squared Error :", round(rmse, 2))

print("R2 Score :", round(r2, 2))


# ============================================================
# 18. ACTUAL VS PREDICTED
# ============================================================

result = pd.DataFrame({
    "Actual BoxOffice": y_test.values,
    "Predicted BoxOffice": y_pred
})

print("\n================ ACTUAL VS PREDICTED ================\n")

print(result)


plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    s=100
)

plt.xlabel("Actual Box Office")

plt.ylabel("Predicted Box Office")

plt.title("Actual vs Predicted Box Office")

# Perfect prediction line

minimum = min(y_test.min(), y_pred.min())

maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.tight_layout()

plt.show()


# ============================================================
# 19. REGRESSION COEFFICIENTS
# ============================================================

print("\n================ MODEL COEFFICIENTS ================\n")

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print(coefficients)


# ============================================================
# 20. PREDICT A NEW MOVIE
# ============================================================

print("\n================ NEW MOVIE PREDICTION ================\n")

# Example movie information

new_movie = pd.DataFrame({
    "Rating": [8.5],
    "Votes": [1500000],
    "Budget": [180],
    "Sentiment_Score": [0.75]
})


predicted_boxoffice = model.predict(
    new_movie
)[0]


print(
    "Predicted Box Office:",
    round(predicted_boxoffice, 2)
)


# Determine success

if predicted_boxoffice >= 800:

    print("Prediction: SUCCESSFUL MOVIE")

else:

    print("Prediction: NOT HIGH BOX OFFICE")


# ============================================================
# 21. EXPORT RESULTS
# ============================================================

df.to_csv(
    "movie_sentiment_results.csv",
    index=False
)

result.to_csv(
    "prediction_results.csv",
    index=False
)

coefficients.to_csv(
    "model_coefficients.csv",
    index=False
)

genre_sentiment.to_csv(
    "genre_sentiment.csv"
)


print("\n================ FILES CREATED ================\n")

print("movie_sentiment_results.csv")

print("prediction_results.csv")

print("model_coefficients.csv")

print("genre_sentiment.csv")


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n==============================================")

print("MOVIE SUCCESS PREDICTION - FINAL SUMMARY")

print("==============================================")

print(
    "Total Movies:",
    len(df)
)

print(
    "Positive Reviews:",
    sum(df["Sentiment"] == "Positive")
)

print(
    "Negative Reviews:",
    sum(df["Sentiment"] == "Negative")
)

print(
    "Neutral Reviews:",
    sum(df["Sentiment"] == "Neutral")
)

print(
    "Model R2 Score:",
    round(r2, 2)
)

print(
    "Average Box Office:",
    round(df["BoxOffice"].mean(), 2)
)

print(
    "Average IMDb Rating:",
    round(df["Rating"].mean(), 2)
)

print("\nProject completed successfully!")
