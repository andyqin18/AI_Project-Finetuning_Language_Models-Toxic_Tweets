import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

st.title("Sentiment Analysis App - beta")
st.header("This app is to analyze the sentiments behind a text. Currently it uses \
          pre-trained models without fine-tuning.")

st.text_input("Enter your text:", value="Missing Sophie.Z...")
st.selectbox("Please select a model:" ("Model 1", "Model 2", "Model 3"))

if st.button("Analyze"):
    st.write("You clicked a button.")
else:
    st.write("Go on! Try the app!")