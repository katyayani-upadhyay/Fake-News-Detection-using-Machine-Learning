# %% [markdown]
# <a href="https://colab.research.google.com/github/katyayani-upadhyay/Fake-News-Detection-using-Machine-Learning/blob/main/Fake_News_Detection_using_Machine_Learning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import streamlit as st
import pandas as pd
import pickle
import os

# Title
st.set_page_config(page_title="Fake News Detection App", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("""
This app predicts whether the news entered is **Fake** or **Real** using a trained Machine Learning model.
""")

# File paths
data_path = "dataset/fake_news.csv"
model_path = "models/model.pkl"
vectorizer_path = "models/vectorizer.pkl"

# Load Data
data = None
if os.path.exists(data_path):
    try:
        data = pd.read_csv(data_path)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        for col in ["title", "subject", "date", "index"]:
            if col in data.columns:
                data.drop(col, axis=1, inplace=True)
        st.sidebar.success("Dataset loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Dataset error: {e}")
else:
    st.sidebar.warning("Dataset not found.")

# Load Model and Vectorizer
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    st.sidebar.success("Model and vectorizer loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Model/vectorizer loading error: {e}")
    st.stop()

# Input text
st.subheader("Enter News Text")
user_input = st.text_area("Paste your news content below:", height=200)

if st.button("Check News Authenticity"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news content to analyze.")
    else:
        try:
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)

            if prediction[0] == 0:
                st.success("✅ This news appears to be **Real**.")
            else:
                st.error("🚫 This news appears to be **Fake**.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

