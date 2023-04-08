import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

st.title("Sentiment Analysis App - beta")
st.header("This app is to analyze the sentiments behind a text. Currently it uses \
          pre-trained models without fine-tuning.")

user_input = st.text_input("Enter your text:", value="Missing Sophie.Z...")
st.selectbox("Please select a model:", ("Model 1", "Model 2", "Model 3"))



if st.button("Analyze"):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    res = classifier(user_input)
    st.write(res)
    
else:
    st.write("Go on! Try the app!")