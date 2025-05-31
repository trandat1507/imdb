import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# =========== 1️⃣ Load IMDb Dataset ===========
print("🔄 Đang load dữ liệu IMDb...")
titles_df = pd.read_csv("title.basics.tsv", sep='\t', na_values='\\N')
ratings_df = pd.read_csv("title.ratings.tsv", sep='\t')

# Lọc phim & merge dữ liệu
movies_df = pd.merge(titles_df, ratings_df, on='tconst')
movies_df = movies_df[
    (movies_df['titleType'] == 'movie') & (movies_df['averageRating'].notnull())
][['primaryTitle', 'genres', 'startYear', 'averageRating', 'numVotes']]
print("✅ Đã load dữ liệu!")

# =========== 2️⃣ Load sample reviews để train model ===========
print("🔄 Đang load dữ liệu sample reviews...")
reviews_df = pd.read_csv("imdb_reviews.csv")

X_train = reviews_df["review"]
y_train = reviews_df["label"]  # Đã đổi thành 'label'
print("✅ Đã load dữ liệu reviews!")

# =========== 3️⃣ Train mô hình ML ===========
print("🔧 Đang train mô hình ML...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("✅ Đã train mô hình!")

# =========== 4️⃣ Lưu mô hình và vectorizer ===========
joblib.dump(model, "sentiment_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
print("✅ Đã lưu mô hình và vectorizer!")

# =========== 5️⃣ Hàm gợi ý phim ===========
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

# =========== 6️⃣ Chạy chương trình ===========
print("\n🎬 Chào mừng đến với Hệ thống Gợi ý phim!")
print("💡 Bạn có thể nhập 1️⃣ review phim hoặc 2️⃣ tâm trạng trực tiếp để nhận gợi ý.\n")

while True:
    mode = input("👉 Nhập 'review' để dự đoán từ nhận xét, 'mood' để nhập cảm xúc, hoặc 'exit' để thoát: ").lower()
    if mode == 'exit':
        print("Tạm biệt! 🎬")
        break
    elif mode == 'review':
        review_text = input("✏️ Nhập nhận xét phim của bạn: ")
        # Dự đoán cảm xúc
        X_input = vectorizer.transform([review_text])
        pred = model.predict(X_input)[0]
        print(f"🔍 Dự đoán cảm xúc của bạn: {pred}")

        # Gợi ý phim
        suggestions = recommend_movies(pred, movies_df, num_recommend=10)
        print("\n🎥 Gợi ý phim dựa trên cảm xúc của bạn:")
        print(suggestions)
    elif mode == 'mood':
        mood = input("🧠 Hôm nay bạn đang có tâm trạng gì (vui, buồn, hành động, kinh dị, tình cảm...)? ")
        suggestions = recommend_movies(mood, movies_df, num_recommend=10)
        print("\n🎥 Gợi ý phim dựa trên tâm trạng của bạn:")
        print(suggestions)
    else:
        print("⚠️ Lựa chọn không hợp lệ! Vui lòng thử lại.\n")
