import streamlit as st
import joblib
import re

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

# ----------------------------
# Cache Model Loading
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_analysis_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# Text Preprocessing
# ----------------------------
def preprocess(text):

    text = text.lower()

    contractions = {
        "don't": "do not",
        "didn't": "did not",
        "doesn't": "does not",
        "can't": "can not",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "mustn't": "must not",
        "needn't": "need not",
        "n't": " not"
    }

    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📌 About")

st.sidebar.info(
    """
**Model:** Linear SVM

**Vectorizer:** TF-IDF

**Accuracy:** 92.78%

Built using Streamlit & Scikit-learn.
"""
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1E88E5;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:25px;
}

.result{
    padding:18px;
    border-radius:12px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:30px;
}

textarea{
    font-size:18px !important;
}

.stButton>button{
    width:100%;
    height:55px;
    font-size:20px;
    border-radius:10px;
    background-color:#1E88E5;
    color:white;
}

.stButton>button:hover{
    background-color:#1565C0;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.markdown(
    "<div class='title'>🤖 AI Sentiment Analysis</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Predict whether a sentence is Positive 😊, Neutral 😐 or Negative 😞</div>",
    unsafe_allow_html=True
)

# ----------------------------
# Sample Inputs
# ----------------------------
st.markdown("### 💡 Try these examples")

st.code("I absolutely loved this movie.")
st.code("The meeting starts at 10 AM.")
st.code("This movie is bad.")

# ----------------------------
# Input
# ----------------------------
text = st.text_area(
    "✍ Enter your text",
    placeholder="Example: I absolutely loved this product!"
)

col1, col2 = st.columns(2)

predict = col1.button("🚀 Predict")
clear = col2.button("🗑 Clear")

# ----------------------------
# Prediction
# ----------------------------
if predict:

    if text.strip() == "":
        st.warning("⚠ Please enter some text.")

    else:

        processed_text = preprocess(text)

        with st.expander("🔍 View Preprocessed Text"):
            st.write(processed_text)

        with st.spinner("Analyzing sentiment..."):

            vector = vectorizer.transform([processed_text])

            prediction = model.predict(vector)[0]

        if prediction == 1:
            st.markdown(
                """
                <div class='result'
                style='background:#d4edda;color:#155724;'>
                😊 Positive Sentiment
                </div>
                """,
                unsafe_allow_html=True
            )

        elif prediction == 0:
            st.markdown(
                """
                <div class='result'
                style='background:#fff3cd;color:#856404;'>
                😐 Neutral Sentiment
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                """
                <div class='result'
                style='background:#f8d7da;color:#721c24;'>
                😞 Negative Sentiment
                </div>
                """,
                unsafe_allow_html=True
            )

# ----------------------------
# Clear Button
# ----------------------------
if clear:
    st.rerun()

# ----------------------------
# Footer
# ----------------------------
st.divider()

st.markdown(
"""
<div class='footer'>
Built with ❤️ using Streamlit | Machine Learning | TF-IDF | Linear SVM
</div>
""",
unsafe_allow_html=True
)