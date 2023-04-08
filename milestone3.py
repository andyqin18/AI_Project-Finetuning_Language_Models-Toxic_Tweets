from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
res = classifier(["I am very happy now.", "Not happy now."])

for result in res:
    print(result)

tokens = tokenizer.tokenize("I am very happy now.")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("I am very happy now.")

print(f'Tokens:{tokens}')
print(f'TokenIDs:{token_ids}')
print(f'InputIDs:{input_ids}')
