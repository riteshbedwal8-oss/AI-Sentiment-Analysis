import re
import pickle
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download('stopwords')

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: #00FFD1;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #CFCFCF;
    margin-bottom: 40px;
}

.stTextArea textarea {
    background-color: #262730;
    color: white;
    border-radius: 15px;
    font-size: 20px;
    padding: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-top: 25px;
    animation: fadeIn 1s;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

.live-box {
    background-color: #1C1F26;
    padding: 15px;
    border-radius: 12px;
    margin-top: 15px;
    font-size: 20px;
    color: #00FFD1;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

with open('models/logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/tfidf_model.pkl', 'rb') as vector_file:
    vectorizer = pickle.load(vector_file)

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

def predict_sentiment(text):

    cleaned_text = clean_text(text)

    neutral_words = [
        "okay",
        "ok",
        "fine",
        "average",
        "normal",
        "nothing special",
        "not bad",
        "not good"
    ]

    for word in neutral_words:

        if word in text.lower():

            return "Neutral 😐", 0.55

    vector_input = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector_input)[0]

    probability = model.predict_proba(vector_input)[0]

    confidence = max(probability)

    if confidence < 0.60:

        return "Neutral 😐", confidence

    elif prediction == 0:

        return "Negative 😡", confidence

    else:

        return "Positive 😊", confidence

st.markdown(
    '<div class="title">🤖 AI Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-Time Emotion Detection using NLP & Machine Learning</div>',
    unsafe_allow_html=True
)

user_input = st.text_area(
    "✍ Type your text below:",
    height=180,
    placeholder="Example: I love this product"
)

if user_input.strip() != "":

    st.markdown(
        """
        <div class="live-box">
            🔍 AI is analyzing your emotions in real time...
        </div>
        """,
        unsafe_allow_html=True
    )

    result, confidence = predict_sentiment(user_input)

    # Positive
    if "Positive" in result:

        st.markdown(
            f"""
            <div class="result-box" style="background-color:#0D825D;color:white;">
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Negative
    elif "Negative" in result:

        st.markdown(
            f"""
            <div class="result-box" style="background-color:#B00020;color:white;">
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Neutral
    else:

        st.markdown(
            f"""
            <div class="result-box" style="background-color:#5A5A5A;color:white;">
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Confidence Bar
    st.progress(float(confidence))

    # Confidence Score
    st.markdown(
        f"""
        <h3 style='text-align:center; color:#00FFD1;'>
            Confidence Score: {confidence:.2%}
        </h3>
        """,
        unsafe_allow_html=True
    )
st.sidebar.title("📌 About Project")

st.sidebar.info("""
Advanced AI Sentiment Analysis Project

Technologies Used:

✅ NLP  
✅ TF-IDF  
✅ Logistic Regression  
✅ Streamlit  
✅ Machine Learning
""")

st.sidebar.title("🧪 Try These Examples")

st.sidebar.success("I love this app")

st.sidebar.error("Worst experience ever")

st.sidebar.warning("The movie was okay")

st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Python, NLP, Machine Learning & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)