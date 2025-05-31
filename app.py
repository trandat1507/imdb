import streamlit as st
import pandas as pd
import joblib

# Load dữ liệu phim
st.write("🔄 Đang load dữ liệu IMDb...")
titles_df = pd.read_csv("title.basics.csv", sep='\t', na_values='\\N')
ratings_df = pd.read_csv("title.ratings.csv", sep='\t')
movies_df = pd.merge(titles_df, ratings_df, on='tconst')
movies_df = movies_df[(movies_df['titleType'] == 'movie') & (movies_df['averageRating'].notnull())]
movies_df = movies_df[['primaryTitle', 'genres', 'startYear', 'averageRating', 'numVotes']]
st.write("✅ Đã load dữ liệu!")

# Load mô hình ML
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Hàm gợi ý phim
def recommend_movies(user_mood, movies_df, num_recommend=10):
    user_mood = user_mood.lower()
    if user_mood in ['vui', 'positive', 'hài hước', 'hạnh phúc']:
        recommended = movies_df.sort_values(by='averageRating', ascending=False).head(num_recommend)
    elif user_mood in ['buồn', 'negative', 'sad']:
        drama_movies = movies_df[movies_df['genres'].str.contains('Drama', na=False)]
        recommended = drama_movies.sort_values(by='averageRating', ascending=False).head(num_recommend)
    elif user_mood in ['hành động', 'action']:
        action_movies = movies_df[movies_df['genres'].str.contains('Action', na=False)]
        recommended = action_movies.sort_values(by='averageRating', ascending=False).head(num_recommend)
    elif user_mood in ['kinh dị', 'horror']:
        horror_movies = movies_df[movies_df['genres'].str.contains('Horror', na=False)]
        recommended = horror_movies.sort_values(by='averageRating', ascending=False).head(num_recommend)
    elif user_mood in ['tình cảm', 'romance']:
        romance_movies = movies_df[movies_df['genres'].str.contains('Romance', na=False)]
        recommended = romance_movies.sort_values(by='averageRating', ascending=False).head(num_recommend)
    else:
        recommended = movies_df.sort_values(by='averageRating', ascending=False).head(num_recommend)

    return recommended[['primaryTitle', 'genres', 'startYear', 'averageRating']]

# Giao diện web
st.title("🎬 Hệ thống Gợi ý Phim theo Cảm Xúc")
st.write("💡 Bạn có thể nhập nhận xét phim hoặc tâm trạng để nhận gợi ý!")

option = st.radio("Chọn phương thức:", ("Nhập nhận xét phim", "Nhập tâm trạng trực tiếp"))

if option == "Nhập nhận xét phim":
    review_text = st.text_area("✏️ Nhập nhận xét phim:")
    if st.button("📢 Dự đoán & Gợi ý phim"):
        if review_text.strip() != "":
            X_input = vectorizer.transform([review_text])
            pred = model.predict(X_input)[0]
            st.write(f"🔍 Dự đoán cảm xúc: **{pred}**")
            suggestions = recommend_movies(pred, movies_df, num_recommend=10)
            st.write("🎥 Gợi ý phim:")
            st.dataframe(suggestions)
        else:
            st.warning("⚠️ Vui lòng nhập nhận xét phim!")
else:
    mood = st.text_input("🧠 Hôm nay bạn đang có tâm trạng gì (vui, buồn, hành động, kinh dị, tình cảm...)?")
    if st.button("🎥 Gợi ý phim"):
        if mood.strip() != "":
            suggestions = recommend_movies(mood, movies_df, num_recommend=10)
            st.write("🎥 Gợi ý phim:")
            st.dataframe(suggestions)
        else:
            st.warning("⚠️ Vui lòng nhập tâm trạng của bạn!")

st.write("---")
st.write("✨ Made with ❤️ by Your Team")
