import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Define global variables
FINE_TUNED_MODEL = "andyqin18/finetuned-bert-uncased"
NUM_SAMPLE_TEXT = 10

# Define analyze function
def analyze(model_name: str, text: str, top_k=1) -> dict:
    '''
    Output result of sentiment analysis of a text through a defined model
    '''
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=top_k)
    return classifier(text)

# App title 
st.title("Sentiment Analysis App - Milestone3")
st.write("This app is to analyze the sentiments behind a text.")
st.write("You can choose to use my fine-tuned model or pre-trained models.")

# Model hub
model_descrip = {
    FINE_TUNED_MODEL: "This is a customized BERT-base finetuned model that detects multiple toxicity for a text. \
        Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate",
    "distilbert-base-uncased-finetuned-sst-2-english": "This model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned on SST-2. \
        Labels: POSITIVE; NEGATIVE ",
    "cardiffnlp/twitter-roberta-base-sentiment": "This is a roBERTa-base model trained on ~58M tweets and finetuned for sentiment analysis with the TweetEval benchmark. \
        Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive",
    "finiteautomata/bertweet-base-sentiment-analysis": "Model trained with SemEval 2017 corpus (around ~40k tweets). Base model is BERTweet, a RoBERTa model trained on English tweets.  \
        Labels: POS; NEU; NEG"
}

user_input = st.text_input("Enter your text:", value="I hate NLP. Always lacking GPU.")
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
            # If fine-tuned
            if user_model == FINE_TUNED_MODEL:
                result = analyze(user_model, user_input, top_k=2)  # Top 2 labels with highest score
                result_dict = {
                        "Text": [user_input],
                        "Highest Toxicity Class": [result[0][0]['label']],
                        "Highest Score": [result[0][0]['score']],
                        "Second Highest Toxicity Class": [result[0][1]['label']],
                        "Second Highest Score": [result[0][1]['score']]
                            }
                st.dataframe(pd.DataFrame(result_dict))

                # 10 Sample Table                       
                sample_texts = [
                    "Please stop. If you continue to vandalize Wikipedia, as you did to Homosexuality, you will be blocked from editing.",
                    "knock it off you bloody CWI trot",
                    "No, he is an arrogant, self serving, immature idiot. Get it right.",
                    "to fuck you and ur family",
                    "Search Google, it's listed as 1966 everywhere I've seen, including many PJ related sites.",
                    "That entry made a lot of sense to me. ",
                    "KSchwartz is an annoying person who often smells of rotten fish and burnt animal hair.",
                    "Cool!",
                    "u suck u suck u suck u suck u sucku suck u suck u suck u suck u u suck",
                    "go fuck yourself ...cunt"
                    ]

                init_table_dict = {
                            "Text": [],
                            "Highest Toxicity Class": [],
                            "Highest Score": [],
                            "Second Highest Toxicity Class": [],
                            "Second Highest Score": []
                                }

                for text in sample_texts:
                    result = analyze(FINE_TUNED_MODEL, text[:50], top_k=2)
                    init_table_dict["Text"].append(text[:50])
                    init_table_dict["Highest Toxicity Class"].append(result[0][0]['label'])
                    init_table_dict["Highest Score"].append(result[0][0]['score'])
                    init_table_dict["Second Highest Toxicity Class"].append(result[0][1]['label'])
                    init_table_dict["Second Highest Score"].append(result[0][1]['score'])
                st.dataframe(pd.DataFrame(init_table_dict))
                st.write("( ─ ‿ ‿ ─ )")


            else:
                result = analyze(user_model, user_input)
                st.write("Result:")
                st.write(f"Label: **{result[0]['label']}**")
                st.write(f"Confidence Score: **{result[0]['score']}**")

else:
    st.write("Go on! Try the app!")