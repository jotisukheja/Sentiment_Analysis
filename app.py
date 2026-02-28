import streamlit as st
import pickle
import numpy as np

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Customer Sentiment Classifier",
    layout="wide",
    page_icon="📝"
)

# ------------------ Load Model & Vectorizer ------------------
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("About this App")
    st.write("""
    This app classifies Amazon Alexa customer reviews into **Positive ✅** and **Negative ❌**

    **Tech Stack:**
    - Python, Streamlit
    - TF-IDF Vectorization
    - Logistic Regression
    - SMOTE for class balancing

    **Instructions:**
    - Enter a review in the text area.
    - Click **Analyze Sentiment** to see the prediction and confidence.
    """)

    st.subheader("Try Sample Reviews")
    sample_reviews = [
        "I love this Alexa! Works perfectly and makes my life easier.",
        "Terrible, waste of money.",
        "The device works okay, but sometimes it's slow.",
        "Amazing product, very easy to use.",
        "Not satisfied, it stopped responding frequently."
    ]
    selected_review = st.selectbox("Select a sample review:", sample_reviews)

# ------------------ Main App ------------------
st.title("📝 Customer Feedback Sentiment Classifier")
st.write("Enter customer feedback text below and the model will classify it as **Positive ✅**, **Negative ❌**, or **Neutral ⚪** (low-confidence).")

input_text = st.text_area("Enter feedback here...", height=150, value=selected_review)

# ------------------ Prediction ------------------
if st.button("Analyze Sentiment"):
    if input_text.strip():
        # Transform input
        text_vector = vectorizer.transform([input_text])
        probas = model.predict_proba(text_vector)[0]
        prediction = model.predict(text_vector)[0]
        confidence = probas[prediction]

        # Define neutral threshold
        neutral_threshold = 0.6

        if confidence < neutral_threshold:
            st.info(f"Neutral ⚪  | Confidence: {confidence:.2f}")
        else:
            if prediction == 1:
                st.success(f"Positive ✅  | Confidence: {confidence:.2f}")
            else:
                st.error(f"Negative ❌  | Confidence: {confidence:.2f}")
    else:
        st.warning("Please enter some text!")


