import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load model and vectorizer
with open("logistic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)  
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def predict_sentiment(review):
    processed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([processed_review])
    prediction = model.predict(vectorized_review)[0]
    return '😊 Positive' if prediction == 1 else '😞 Negative'

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8f9fa;
            padding: 20px;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #007acc;
        }
        .custom-textarea {
            width: 100%;
            height: 180px;
            padding: 15px;
            border-radius: 12px;
            border: 2px solid #007acc;
            background-color: white;
            font-size: 16px;
            box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.1);
            resize: none;
            color: black;
        }
        .btn {
            background-color: #007acc;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px;
            font-size: 18px;
            width: 100%;
            text-align: center;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }
        .btn:hover {
            background-color: #005b99;
        }
        .result {
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            border-radius: 10px;
            background-color: #e6f2ff;
            color: #007acc;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">🎬 Movie Review Sentiment Analysis 🎭</h1>', unsafe_allow_html=True)
st.markdown("#### Enter a movie review below to analyze its sentiment.")

user_input = st.text_area("", height=180, key="input_review")

# Predict Button
if st.button("🔍 Analyze Sentiment", help="Click to analyze the sentiment of your review", key="analyze_button"):
    result = predict_sentiment(user_input)
    st.markdown(f'<div class="result">{result}</div>', unsafe_allow_html=True)
