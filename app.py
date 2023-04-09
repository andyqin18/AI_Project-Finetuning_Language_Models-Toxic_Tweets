import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

st.title("Sentiment Analysis App - beta")
st.header("This app is to analyze the sentiments behind a text. Currently it uses \
          pre-trained models without fine-tuning.")

user_input = st.text_input("Enter your text:", value="Missing Sophie.Z...")
user_model = st.selectbox("Please select a model:", 
                          ("distilbert-base-uncased-finetuned-sst-2-english",
                           "cardiffnlp/twitter-roberta-base-sentiment", 
                           "finiteautomata/bertweet-base-sentiment-analysis"))

def analyze(model_name, text):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier(text)


if st.button("Analyze"):
    if not user_input:
        st.write("Please enter a text.")
    else:
        with st.spinner("Hang on.... Analyzing..."):
            st.write(analyze(user_model, user_input))
else:
    st.write("Go on! Try the app!")