# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# def analyze(model_name: str, text: str, top_k=1) -> dict:
#     '''
#     Output result of sentiment analysis of a text through a defined model
#     '''
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=top_k)
#     return classifier(text)


# user_input = "Go fuck yourself"
# user_model = "andyqin18/test-finetuned"

# result = analyze(user_model, user_input, top_k=4)

# print(result[0][0]['label'])

import pandas as pd
import numpy as np
df = pd.read_csv("milestone3/comp/test_comment.csv")
test_texts = df["comment_text"].values
sample_texts = np.random.choice(test_texts, size=10, replace=False)
print(sample_texts)