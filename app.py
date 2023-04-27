import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Define analyze function
def analyze(model_name: str, text: str) -> dict:
    '''
    Output result of sentiment analysis of a text through a defined model
    '''
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier(text)

# App title 
st.title("Sentiment Analysis App - Milestone2")
st.write("This app is to analyze the sentiments behind a text.")
st.write("Currently it uses pre-trained models without fine-tuning.")

# Model hub
model_descrip = {
    "andyqin18/test-finetuned": "This is a customized BERT-base finetuned model that detects multiple toxicity for a text. \
        Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate",
    "distilbert-base-uncased-finetuned-sst-2-english": "This model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned on SST-2. \
        Labels: POSITIVE; NEGATIVE ",
    "cardiffnlp/twitter-roberta-base-sentiment": "This is a roBERTa-base model trained on ~58M tweets and finetuned for sentiment analysis with the TweetEval benchmark. \
        Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive",
    "finiteautomata/bertweet-base-sentiment-analysis": "Model trained with SemEval 2017 corpus (around ~40k tweets). Base model is BERTweet, a RoBERTa model trained on English tweets.  \
        Labels: POS; NEU; NEG"
}

user_input = st.text_input("Enter your text:", value="NYU is the better than Columbia.")
user_model = st.selectbox("Please select a model:", model_descrip)

# Display model information
st.write("### Model Description:")
st.write(model_descrip[user_model])

# Perform analysis and print result
if st.button("Analyze"):
    if not user_input:
        st.write("Please enter a text.")
    else:
        with st.spinner("Hang on.... Analyzing..."):
            result = analyze(user_model, user_input)
            st.write("Result:")
            st.write(f"Label: **{result[0]['label']}**")
            st.write(f"Confidence Score: **{result[0]['score']}**")

else:
    st.write("Go on! Try the app!")