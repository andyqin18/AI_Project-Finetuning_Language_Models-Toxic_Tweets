import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def analyze(model_name, text):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier(text)

st.title("Sentiment Analysis App - beta")
st.write("This app is to analyze the sentiments behind a text. \n Currently it uses \
          pre-trained models without fine-tuning.")

model_descrip = {
    "distilbert-base-uncased-finetuned-sst-2-english": "This model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned on SST-2.\n \
        Labels: POSITIVE; NEGATIVE ",
    "cardiffnlp/twitter-roberta-base-sentiment": "This is a roBERTa-base model trained on ~58M tweets and finetuned for sentiment analysis with the TweetEval benchmark.\n \
        Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive",
    "finiteautomata/bertweet-base-sentiment-analysis": "Model trained with SemEval 2017 corpus (around ~40k tweets). Base model is BERTweet, a RoBERTa model trained on English tweets. \n \
        Labels: POS; NEU; NEG"
}


user_input = st.text_input("Enter your text:", value="Missing Sophie.Z...")
user_model = st.selectbox("Please select a model:", 
                          model_descrip)

st.write("### Model Description:")
st.write(model_descrip[user_model])


if st.button("Analyze"):
    if not user_input:
        st.write("Please enter a text.")
    else:
        with st.spinner("Hang on.... Analyzing..."):
            result = analyze(user_model, user_input)
            st.write(f"Result: \nLabel: {result[0]['label']} Score: {result[0]['score']}")
else:
    st.write("Go on! Try the app!")