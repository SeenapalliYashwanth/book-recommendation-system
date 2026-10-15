
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    return joblib.load("artifacts/book_recommender.pkl")

artifacts = load_artifacts()

svd_model = artifacts["svd_model"]
train_data = artifacts["train_data"]
books = artifacts["books_model"].drop_duplicates("ISBN").copy()
content_books = artifacts["content_books"]
tfidf = artifacts["tfidf_vectorizer"]
popularity = artifacts["popularity_ranking"]

# Normalize identifiers for reliable lookups
books["ISBN"] = books["ISBN"].astype(str)
content_books["ISBN"] = content_books["ISBN"].astype(str)
train_data = train_data.copy()
train_data["ISBN"] = train_data["ISBN"].astype(str)

book_lookup = books.set_index("ISBN", drop=False)
known_users = set(train_data["User-ID"].unique())
known_books = set(train_data["ISBN"].unique())

st.title("📚 Book Recommendation System")
st.write(
    "Discover books using personalized recommendations, "
    "book similarity, and popularity."
)

tab1, tab2, tab3 = st.tabs([
    "Personalized",
    "Similar Books",
    "Popular Books"
])

# ---------------- Personalized recommendations ----------------
with tab1:
    st.subheader("Recommendations for a reader")

    user_options = sorted(known_users)
    selected_user = st.selectbox(
        "Select a user ID",
        user_options,
        key="personal_user"
    )

    top_n = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        key="personal_top_n"
    )

    if st.button("Recommend books", key="personal_button"):
        seen = set(
            train_data.loc[
                train_data["User-ID"] == selected_user,
                "ISBN"
            ]
        )

        candidates = [
            isbn for isbn in known_books
            if isbn not in seen
        ]

        scored = [
            (isbn, svd_model.predict(selected_user, isbn).est)
            for isbn in candidates
        ]

        scored.sort(key=lambda item: item[1], reverse=True)
        results = scored[:top_n]

        if not results:
            st.info("No unseen candidate books were found.")
        else:
            result_df = pd.DataFrame(
                results,
                columns=["ISBN", "Predicted Rating"]
            )
            result_df = result_df.merge(
                books[["ISBN", "Book-Title", "Book-Author"]],
                on="ISBN",
                how="left"
            )

            st.dataframe(
                result_df[
                    ["Book-Title", "Book-Author", "Predicted Rating"]
                ].rename(columns={
                    "Book-Title": "Book",
                    "Book-Author": "Author"
                }),
                use_container_width=True,
                hide_index=True
            )

# ---------------- Similar books ----------------
with tab2:
    st.subheader("Find books similar to a selected title")

    title_options = sorted(
        content_books["Book-Title"].dropna().unique().tolist()
    )

    selected_title = st.selectbox(
        "Choose a book",
        title_options,
        key="similar_title"
    )

    similar_n = st.slider(
        "Number of similar books",
        min_value=5,
        max_value=20,
        value=10,
        key="similar_n"
    )

    if st.button("Find similar books", key="similar_button"):
        matches = content_books[
            content_books["Book-Title"] == selected_title
        ]

        if matches.empty:
            st.warning("Book not found.")
        else:
            selected_row = matches.iloc[0]
            selected_idx = matches.index[0]

            matrix = tfidf.transform(
                content_books["Features"]
            ) if "Features" in content_books.columns else None

            if matrix is None:
                st.error(
                    "Content feature text is missing from the saved data."
                )
            else:
                scores = cosine_similarity(
                    matrix[selected_idx], matrix
                ).ravel()

                scores[selected_idx] = -1
                count = min(similar_n, len(scores) - 1)

                indices = np.argsort(scores)[::-1][:count]
                result = content_books.iloc[indices][
                    ["Book-Title", "Book-Author"]
                ].copy()
                result["Similarity"] = scores[indices]

                st.dataframe(
                    result.rename(columns={
                        "Book-Title": "Book",
                        "Book-Author": "Author"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

# ---------------- Popular books ----------------
with tab3:
    st.subheader("Popular book recommendations")

    popular_n = st.slider(
        "Number of popular books",
        min_value=5,
        max_value=20,
        value=10,
        key="popular_n"
    )

    if st.button("Show popular books", key="popular_button"):
        result = popularity.head(popular_n).copy()

        result["ISBN"] = result["ISBN"].astype(str)
        result = result.merge(
            books[["ISBN", "Book-Title", "Book-Author"]],
            on="ISBN",
            how="left"
        )

        display_columns = [
            col for col in [
                "Book-Title",
                "Book-Author",
                "Weighted_Rating",
                "Rating_Count"
            ]
            if col in result.columns
        ]

        st.dataframe(
            result[display_columns].rename(columns={
                "Book-Title": "Book",
                "Book-Author": "Author"
            }),
            use_container_width=True,
            hide_index=True
        )

st.caption(
    "Recommendations are generated from historical ratings and book metadata. "
    "Predicted ratings are estimates, not guaranteed user ratings."
)
