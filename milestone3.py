from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print(classifier.__class__)
res = classifier(["I am very happy now.", "Not happy now."])

for result in res:
    print(result)

# Separate each word as a token
tokens = tokenizer.tokenize("I am very happy now.")

# Generate a list of IDs, each ID for each token
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Return a dict with IDs
input_ids = tokenizer("I am very happy now.")

print(f'Tokens:{tokens}')
print(f'TokenIDs:{token_ids}')
print(f'InputIDs:{input_ids}')

X_train = ["We are very happy to show you the Transformers library.", 
           "Hope you don't hate it"]

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")


