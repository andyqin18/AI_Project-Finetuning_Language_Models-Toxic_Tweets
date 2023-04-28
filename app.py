import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


fine_tuned_model = "andyqin18/test-finetuned"
sample_text_num = 10

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
    fine_tuned_model: "This is a customized BERT-base finetuned model that detects multiple toxicity for a text. \
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
            if user_model == fine_tuned_model:
                result = analyze(user_model, user_input, top_k=2)
                result_dict = {
                        "Text": [user_input],
                        "Highest Toxicity Class": [result[0][0]['label']],
                        "Highest Score": [result[0][0]['score']],
                        "Second Highest Toxicity Class": [result[0][1]['label']],
                        "Second Highest Score": [result[0][1]['score']]
                            }
                st.dataframe(pd.DataFrame(result_dict))
                if st.button("Click to generate ten sample analysis"):
                    df = pd.read_csv("milestone3/comp/test_comment.csv")
                    test_texts = df["comment_text"].values
                    sample_texts = np.random.choice(test_texts, size=sample_text_num, replace=False)

                    init_table_dict = {
                                "Text": [],
                                "Highest Toxicity Class": [],
                                "Highest Score": [],
                                "Second Highest Toxicity Class": [],
                                "Second Highest Score": []
                                    }

                    for text in sample_texts:
                        result = analyze(fine_tuned_model, text[:50], top_k=2)
                        init_table_dict["Text"].append(text[:50])
                        init_table_dict["Highest Toxicity Class"].append(result[0][0]['label'])
                        init_table_dict["Highest Score"].append(result[0][0]['score'])
                        init_table_dict["Second Highest Toxicity Class"].append(result[0][1]['label'])
                        init_table_dict["Second Highest Score"].append(result[0][1]['score'])
                        st.dataframe(pd.DataFrame(init_table_dict))
                else:
                    st.write("(─‿‿─)")


            else:
                result = analyze(user_model, user_input)
                st.write("Result:")
                st.write(f"Label: **{result[0]['label']}**")
                st.write(f"Confidence Score: **{result[0]['score']}**")

else:
    st.write("Go on! Try the app!")