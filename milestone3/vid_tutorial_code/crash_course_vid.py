from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print(classifier.__class__)
res = classifier(["We are very happy to show you the Transformers library.", 
           "Hope you don't hate it"])

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

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt") # "pt" for PyTorch

# batch = tokenizer(X_train, padding=True, truncation=True, max_length=512)
# batch = torch.tensor(batch["input_ids"])

with torch.no_grad(): # Disable gradient tracking
    outputs = model(**batch) # "**" unpacks values in the dictionary, loss=None
    outputs = model(**batch, labels=torch.tensor([1, 0])) # Now we see the loss
    print("Outputs: ", outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print("Predictions: ", predictions)
    label_ids = torch.argmax(predictions, dim=1)
    print("Raw Labels: ", label_ids)
    labels = [model.config.id2label[label_id] for label_id in label_ids.tolist()]
    print("Labels: ", labels)


# save_directory = "saved"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)

# tokenizer = AutoTokenizer.from_pretrained(save_directory)
# model = AutoModelForSequenceClassification.from_pretrained(save_directory)



