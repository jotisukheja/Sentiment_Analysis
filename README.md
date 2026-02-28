# Sentiment Analysis

A web application that analyzes Amazon Alexa reviews and predicts sentiment — positive or negative — using a machine learning model trained from scratch.

**Live Demo:** [alexasentimentanalysis.streamlit.app](https://alexasentimentanalysis.streamlit.app/)  
**Source Code:** [GitHub Repository](https://github.com/jotisukheja/Sentiment_Analysis)

---

## Overview

This project processes Amazon Alexa user reviews and classifies them as positive or negative. The machine learning model is trained from scratch using a **TF-IDF vectorizer** and a **classification algorithm**. Results are displayed in an interactive interface built with **Streamlit**, making it easy to explore predictions in real time.

---

## Built With

- **Python** – core programming language  
- **Streamlit** – for interactive web interface  
- **Scikit-learn** – model training and prediction  
- **Pandas & NumPy** – data handling and preprocessing  
- **Git LFS** – for storing model files  

---

## Project Structure
Sentiment_Analysis/
│
├── app.py # Streamlit application     
├── requirements.txt # Python dependencies   
├── .gitignore   
├── data/   
│ └── amazon_alexa.tsv # Dataset containing reviews and sentiments   
├── sentiment_model.pkl # Trained classification model   
├── tfidf_vectorizer.pkl # TF-IDF vectorizer trained on the dataset   
└── Alexa_Sentiment_Analysis.ipynb # Notebook with data exploration and model training 



---

## How It Works

1. Load the dataset containing Amazon Alexa reviews.  
2. Preprocess the text data and transform it using the **TF-IDF vectorizer** trained on the dataset.  
3. Train a **classification model** from scratch to predict sentiment (positive/negative).  
4. Display predictions and results interactively using the **Streamlit app**.

---


## How to Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/jotisukheja/Sentiment_Analysis.git
cd Sentiment_Analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app
```bash
streamlit run app.py
```
- Open your browser at http://localhost:8501 to access the application.
