from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def analyze(model_name: str, text: str, top_k=1) -> dict:
    '''
    Output result of sentiment analysis of a text through a defined model
    '''
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=top_k)
    return classifier(text)


user_input = "Go fuck yourself"
user_model = "andyqin18/test-finetuned"

# result = analyze(user_model, user_input, top_k=2)

# print(result[0][0]['label'])

import pandas as pd
import numpy as np
df = pd.read_csv("milestone3/comp/test_comment.csv")
test_texts = df["comment_text"].values
sample_texts = np.random.choice(test_texts, size=10, replace=False)
init_table_dict = {
            "Text": [],
            "Highest Toxicity Class": [],
            "Highest Score": [],
            "Second Highest Toxicity Class": [],
            "Second Highest Score": []
                }

for text in sample_texts:
    result = analyze(user_model, text, top_k=2)
    init_table_dict["Text"].append(text[:50])
    init_table_dict["Highest Toxicity Class"].append(result[0][0]['label'])
    init_table_dict["Highest Score"].append(result[0][0]['score'])
    init_table_dict["Second Highest Toxicity Class"].append(result[0][1]['label'])
    init_table_dict["Second Highest Score"].append(result[0][1]['score'])

print(init_table_dict)